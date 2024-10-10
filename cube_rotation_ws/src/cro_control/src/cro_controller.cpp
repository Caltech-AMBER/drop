#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "mjpc/tasks/leap/leap.h"
#include "obelisk_ros_utils.h"
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <glfw_adapter.h>
#include <mjpc/agent.h>
#include <mjpc/app.h>
#include <mjpc/array_safety.h>
#include <mjpc/planners/planner.h>
#include <mjpc/simulate.h>
#include <mjpc/task.h>
#include <mjpc/tasks/tasks.h>
#include <mjpc/threadpool.h>
#include <mjpc/utilities.h>
#include <mujoco/mujoco.h>
#include <obelisk_control_msgs/msg/position_setpoint.hpp>
#include <obelisk_controller.h>
#include <obelisk_estimator_msgs/msg/estimated_state.hpp>
#include <obelisk_sensor_msgs/msg/obk_joint_encoders.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

// absl flags
ABSL_FLAG(std::string, task, "Leap", "Which model to load on startup.");
ABSL_FLAG(bool, planner_enabled, true, "If true, the planner will run on startup");
ABSL_FLAG(float, sim_percent_realtime, 100, "The realtime percentage at which the simulation will be launched.");
ABSL_FLAG(bool, estimator_enabled, false, "If true, estimator loop will run on startup");
ABSL_FLAG(bool, show_left_ui, true, "If true, the left UI (ui0) will be visible on startup");
ABSL_FLAG(bool, show_plot, true, "If true, the plots will be visible on startup");
ABSL_FLAG(bool, show_info, true, "If true, the infotext panel will be visible on startup");

// [NOTE] these are defined up here because they need to be exposed to the
// mujoco callbacks, and we can't use extern in a class context (need
// to expose the callbacks to mujoco's C engine via the variables
// mjcb_controller and mjcb_sensor)
mjModel* m = nullptr;
mjData* d  = nullptr;
std::unique_ptr<mujoco::Simulate> sim;

extern "C" {
void controller(const mjModel* m, mjData* d);
}

/**
 * @brief MJPC controller callback to process control logic within the simulation.
 *
 * @param m Pointer to the MuJoCo model structure.
 * @param data Pointer to the MuJoCo data structure.
 */
void controller(const mjModel* m, mjData* data) {
    // if agent, skip
    if (data != d) {
        return;
    }

    // if simulation:
    if (sim->agent->action_enabled) {
        sim->agent->ActivePlanner().ActionFromPolicy(data->ctrl, &sim->agent->state.state()[0],
                                                     sim->agent->state.time());
    }
}

extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

/**
 * @brief MJPC sensor callback to process sensor data at a specific simulation stage.
 *
 * @param model Pointer to the MuJoCo model structure.
 * @param data Pointer to the MuJoCo data structure.
 * @param stage The current stage of sensor processing.
 */
// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
    if (stage == mjSTAGE_ACC) {
        if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0) {
            if (sim->agent->IsPlanningModel(model)) {
                // the planning thread and rollout threads don't need
                // synchronization when using PlanningResidual.
                const mjpc::ResidualFn* residual = sim->agent->PlanningResidual();
                residual->Residual(model, data, data->sensordata);
            } else {
                // this residual is used by the physics thread and the UI thread (for
                // plots), and is run with a shared lock, to safely run with changes to
                // weights and parameters
                sim->agent->ActiveTask()->Residual(model, data, data->sensordata);
            }
        }
    }
}

// type aliases for readability
using CallbackReturn    = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using PositionSetpoint  = obelisk_control_msgs::msg::PositionSetpoint;
using EstimatedState    = obelisk_estimator_msgs::msg::EstimatedState;
using ObeliskController = obelisk::ObeliskController<PositionSetpoint, EstimatedState>;

/**
 * @brief The cube rotation obelisk controller class.
 */
class CroController : public ObeliskController {
  public:
    CroController(const std::string& node_name) : ObeliskController(node_name) {
        this->declare_parameter("mjpc_render", true);

        this->declare_parameter("sampling_trajectories", 120);
        this->declare_parameter("agent_planner", 5); // the type of planner, 5=CEM, 0=Pred Sampling
        this->declare_parameter("agent_horizon", 0.5);
        this->declare_parameter("agent_timestep", 0.01);
        this->declare_parameter("n_elite", 5);
        this->declare_parameter("sampling_spline_points", 4);
        this->declare_parameter("sampling_representation", 0); // 0 for zero-order, 1 for linear, 2 for cubic

        this->declare_parameter("cost_pos", 200.0);
        this->declare_parameter("cost_orientation", 1.0);
        this->declare_parameter("cost_actuation", 0.0);
        this->declare_parameter("cost_grasp", 0.0);
        this->declare_parameter("cost_cube_linear_velocity", 0.05);
        this->declare_parameter("cost_cube_angular_velocity", 0.025);

        this->declare_parameter("std_min", 0.5);              // min noise for CEM
        this->declare_parameter("sampling_exploration", 0.3); // noise for predictive sampling
        this->declare_parameter("axis_aligned_goal", 1);

        // subscriber for the estimated state of the cube
        this->RegisterObkSubscription<EstimatedState>(
            "sub_q_cube_setting", sub_q_cube_key_, std::bind(&CroController::UpdateXCube, this, std::placeholders::_1));

        // timer/publisher pair for publishing cube goal (generated internally by the planner)
        // TODO(ahl): expose the cube goal to the planner potentially
        this->RegisterObkTimer("timer_q_cube_goal_setting", timer_q_cube_goal_key_,
                               std::bind(&CroController::CubeGoalTimerCallback, this));
        this->RegisterObkPublisher<EstimatedState>("pub_q_cube_goal_setting", pub_q_cube_goal_key_);
    }

  private:
    // constants
    const int kQCubeDim = 7;  // cube pose (x, y, z, qw, qx, qy, qz)
    const int kVCubeDim = 6;  // cube velocity (vx, vy, vz, wx, wy, wz)
    const int kQLeapDim = 16; // leap configuration
    const int kVLeapDim = 16; // leap velocity
    const int kQDim     = kQCubeDim + kQLeapDim;
    const int kVDim     = kVCubeDim + kVLeapDim;

    const std::string timer_q_cube_goal_key_ = "timer_q_cube_goal";
    const std::string pub_q_cube_goal_key_   = "pub_q_cube_goal";
    const std::string sub_q_cube_key_        = "sub_q_cube";

    // mjpc variables
    bool sim_ready = false;

    // thread just to call initializeMJPC
    std::thread initialize_mjpc_thread_;

    // internal state estimates
    std::vector<double> q_cube_hat_;
    std::vector<double> q_leap_hat_;
    std::vector<double> v_cube_hat_;
    std::vector<double> v_leap_hat_;

    // global reference time
    double global_time = this->get_clock()->now().seconds();

    // ////////////////////////// //
    // STATE TRANSITION CALLBACKS //
    // ////////////////////////// //

    /**
     * @brief Lifecycle node interface callback for the configure transition.
     *
     * @param state The current state of the lifecycle node.
     * @return The callback return status.
     */
    CallbackReturn on_configure(const rclcpp_lifecycle::State& state) {
        ObeliskController::on_configure(state);

        // initialize mjpc
        initialize_mjpc_thread_ = std::thread(&CroController::initializeMJPC, this);
        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Lifecycle node interface callback for the activate transition.
     *
     * @param state The current state of the lifecycle node.
     * @return The callback return status.
     */
    CallbackReturn on_activate(const rclcpp_lifecycle::State& state) {
        ObeliskController::on_activate(state);
        sim_ready = true;
        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Lifecycle node interface callback for the deactivate transition.
     *
     * @param state The current state of the lifecycle node.
     * @return The callback return status.
     */
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) {
        ObeliskController::on_deactivate(state);
        sim_ready = false;
        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Lifecycle node interface callback for the cleanup transition.
     *
     * @param state The current state of the lifecycle node.
     * @return The callback return status.
     */
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) {
        ObeliskController::on_cleanup(state);
        sim->exitrequest = true;
        if (initialize_mjpc_thread_.joinable()) {
            initialize_mjpc_thread_.join();
        }
        return CallbackReturn::SUCCESS;
    }

    // ///////////////// //
    // OBELISK CALLBACKS //
    // ///////////////// //

    /**
     * @brief Update the estimated LEAP hand state based on the received message.
     *
     * We only update the LEAP hand states here, as the cube state is updated by a separate callback.
     *
     * @param msg The estimated state message received from the estimator.
     */
    void UpdateXHat(const EstimatedState& msg) override {
        if (m && d && sim && sim_ready) {
            q_leap_hat_ = msg.q_joints;
            v_leap_hat_ = msg.v_joints;
            mju_copy(d->qpos + kQCubeDim, q_leap_hat_.data(), kQLeapDim);
            mju_copy(d->qvel + kVCubeDim, v_leap_hat_.data(), kVLeapDim);
            setState(m, d);
        }
    }

    /**
     * @brief Update the estimated cube state based on the received message.
     *
     * @param msg The estimated state message received from the estimator.
     */
    void UpdateXCube(const EstimatedState& msg) {
        if (m && d && sim && sim_ready) {
            q_cube_hat_ = msg.q_base;
            v_cube_hat_ = msg.v_base;
            mju_copy(d->qpos, q_cube_hat_.data(), kQCubeDim);
            mju_copy(d->qvel, v_cube_hat_.data(), kVCubeDim);
            setState(m, d);
        }
    }

    /**
     * @brief Compute and return the control action to be applied.
     *
     * @return The position setpoint message containing the computed control action.
     */
    PositionSetpoint ComputeControl() override {
        if (m && d && sim && sim_ready) {
            PositionSetpoint msg;

            // querying action from the planner
            std::vector<double> ctrl(m->nu, 0.0);
            sim->agent->ActivePlanner().ActionFromPolicy(ctrl.data(), sim->agent->state.state().data(),
                                                         sim->agent->state.time());

            // publishing the action
            msg.u_mujoco = ctrl;
            msg.q_des    = ctrl;
            this->GetPublisher<PositionSetpoint>(this->ctrl_key_)->publish(msg);
            return msg;
        }
    }

    /**
     * @brief Timer callback to publish cube goal visualization data.
     */
    void CubeGoalTimerCallback() {
        if (m && d && sim && sim_ready) {
            EstimatedState msg;
            std::vector<double> q_cube_goal(7, 0.0);
            q_cube_goal[0] = 0.1;
            q_cube_goal[1] = -0.2;
            q_cube_goal[2] = 0.05;
            mju_copy(q_cube_goal.data() + 3, d->mocap_quat, 4);
            msg.q_base         = q_cube_goal;
            msg.base_link_name = "cube_goal";
            this->GetPublisher<EstimatedState>(this->pub_q_cube_goal_key_)->publish(msg);
        }
    }

    // ////////// //
    // MJPC UTILS //
    // ////////// //

    /**
     * @brief Load the MJPC model using the given agent and simulation context.
     *
     * @param agent Pointer to the MJPC agent.
     * @param sim Reference to the MuJoCo simulation object.
     * @return Pointer to the newly loaded MJPC model.
     */
    mjModel* LoadModel(const mjpc::Agent* agent, mujoco::Simulate& sim) {
        mjpc::Agent::LoadModelResult load_model = sim.agent->LoadModel("");
        mjModel* mnew                           = load_model.model.release();
        mujoco::util_mjpc::strcpy_arr(sim.load_error, load_model.error.c_str());
        if (!mnew) {
            RCLCPP_ERROR_STREAM(this->get_logger(), load_model.error);
            return nullptr;
        }
        return mnew;
    }

    /**
     * @brief Set the state of the MJPC simulation with the given model and data.
     *
     * @param mnew Pointer to the new MuJoCo model.
     * @param dnew Pointer to the new MuJoCo data.
     */
    void setState(mjModel* mnew, mjData* dnew) {
        mj_forward(mnew, dnew);                               // updates the data struct
        if (sim_ready) {
            sim->agent->ActiveTask()->Transition(mnew, dnew); // checks if cube should reset
        }
        double time = this->get_clock()->now().seconds() - global_time;
        dnew->time  = time;
        sim->agent->state.Set(mnew, dnew); // sets the agent's state to match the GUI
    }

    /**
     * @brief Initialize the MJPC simulation environment and setup necessary components.
     */
    void initializeMJPC() {
        using namespace mjpc;

        // only create the GLFW adapter if we're rendering
        bool mjpc_render = this->get_parameter("mjpc_render").as_bool();
        std::unique_ptr<mujoco::PlatformUIAdapter> adapter =
            mjpc_render ? std::make_unique<mujoco::GlfwAdapter>() : nullptr;

        // boilerplate sim startup
        std::vector<std::shared_ptr<Task>> tasks = {std::make_shared<Leap>()};
        int task_id                              = 0;
        sim = std::make_unique<mujoco::Simulate>(std::move(adapter), std::make_shared<Agent>());
        sim->agent->SetTaskList(std::move(tasks));
        sim->agent->gui_task_id = task_id;
        sim->filename           = sim->agent->GetTaskXmlPath(sim->agent->gui_task_id);
        m                       = LoadModel(sim->agent.get(), *sim);
        d                       = mj_makeData(m);
        sim->mnew               = m;
        sim->dnew               = d;

        // configure parameters of mjpc
        SetSamplingTrajectories(m, this->get_parameter("sampling_trajectories").as_int());
        SetAgentPlanner(m, this->get_parameter("agent_planner").as_int());
        SetAgentHorizon(m, this->get_parameter("agent_horizon").as_double());
        SetAgentTimestep(m, this->get_parameter("agent_timestep").as_double());
        SetNElite(m, this->get_parameter("n_elite").as_int());
        SetSamplingSplinePoints(m, this->get_parameter("sampling_spline_points").as_int());
        SetSamplingRepresentation(m, this->get_parameter("sampling_representation").as_int());
        SetCostPos(m, this->get_parameter("cost_pos").as_double());
        SetCostOrientation(m, this->get_parameter("cost_orientation").as_double());
        SetCostActuation(m, this->get_parameter("cost_actuation").as_double());
        SetCostGrasp(m, this->get_parameter("cost_grasp").as_double());
        SetCostCubeLinearVelocity(m, this->get_parameter("cost_cube_linear_velocity").as_double());
        SetCostCubeAngularVelocity(m, this->get_parameter("cost_cube_angular_velocity").as_double());
        SetMinStd(m, this->get_parameter("std_min").as_double());
        SetSamplingExploration(m, this->get_parameter("sampling_exploration").as_double());
        SetAxisAlignedGoal(m, this->get_parameter("axis_aligned_goal").as_int());

        // agent
        sim->agent->estimator_enabled = absl::GetFlag(FLAGS_estimator_enabled);
        sim->agent->Initialize(m);
        sim->agent->Allocate();
        sim->agent->Reset();
        int home_id = mj_name2id(m, mjOBJ_KEY, "home");
        if (home_id >= 0) {
            mj_resetDataKeyframe(m, d, home_id);
            sim->agent->Reset(d->ctrl);
        } else {
            sim->agent->Reset();
        }
        sim->agent->PlotInitialize();
        sim->agent->PlotReset();
        sim->agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);

        // other sim settings
        sim->real_time_index = 100;
        sim->delete_old_m_d  = true;
        sim->ui0_enable      = mjpc_render ? absl::GetFlag(FLAGS_show_left_ui) : false;
        sim->info            = mjpc_render ? absl::GetFlag(FLAGS_show_info) : false;

        // updating the GUI/agent
        sim->loadrequest = 2;                       // loads the task
        sim->agent->ActiveTask()->Transition(m, d); // causes robot to appear
        setState(m, d);                             // set initial state

        // setting global mujoco callbacks - necessary to update residual
        mjcb_control = controller;
        mjcb_sensor  = sensor;

        if (mjpc_render) {
            // start the planning and render loop
            sim->InitializeRenderLoop();
            {
                mjpc::ThreadPool plan_pool(1);
                plan_pool.Schedule([this]() { sim->agent->Plan(sim->exitrequest, sim->uiloadrequest); }); // forked
                sim->RenderLoop(); // blocking call!
            }
        } else {
            // start the planning loop only
            mjpc::ThreadPool plan_pool(1);
            plan_pool.Schedule([this]() { sim->agent->Plan(sim->exitrequest, sim->uiloadrequest); });
        }
    }

    /**
     * @brief Set the number of sampling trajectories for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param sampling_trajectories The number of sampling trajectories to set.
     */
    void SetSamplingTrajectories(mjModel* m, int sampling_trajectories) {
        int id_sampling_trajectories = mj_name2id(m, mjOBJ_NUMERIC, "sampling_trajectories");
        m->numeric_data[m->numeric_adr[id_sampling_trajectories]] = sampling_trajectories;
        RCLCPP_INFO_STREAM(this->get_logger(), "sampling_trajectories=" << sampling_trajectories);
    }

    /**
     * @brief Set the agent planner for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param planner The planner to set.
     */
    void SetAgentPlanner(mjModel* m, int planner) {
        int id_planner                              = mj_name2id(m, mjOBJ_NUMERIC, "agent_planner");
        m->numeric_data[m->numeric_adr[id_planner]] = planner;
        RCLCPP_INFO_STREAM(this->get_logger(), "agent_planner=" << planner);
    }

    /**
     * @brief Set the agent horizon for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param horizon The horizon to set.
     */
    void SetAgentHorizon(mjModel* m, double horizon) {
        int id_horizon                              = mj_name2id(m, mjOBJ_NUMERIC, "agent_horizon");
        m->numeric_data[m->numeric_adr[id_horizon]] = horizon;
        RCLCPP_INFO_STREAM(this->get_logger(), "agent_horizon=" << horizon);
    }

    /**
     * @brief Set the agent timestep for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param timestep The timestep to set.
     */
    void SetAgentTimestep(mjModel* m, double timestep) {
        int id_timestep                              = mj_name2id(m, mjOBJ_NUMERIC, "agent_timestep");
        m->numeric_data[m->numeric_adr[id_timestep]] = timestep;
        RCLCPP_INFO_STREAM(this->get_logger(), "agent_timestep=" << timestep);
    }

    /**
     * @brief Set the number of elite samples for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param n_elite The number of elite samples to set.
     */
    void SetNElite(mjModel* m, int n_elite) {
        int id_n_elite                              = mj_name2id(m, mjOBJ_NUMERIC, "n_elite");
        m->numeric_data[m->numeric_adr[id_n_elite]] = n_elite;
        RCLCPP_INFO_STREAM(this->get_logger(), "n_elite=" << n_elite);
    }

    /**
     * @brief Set the number of spline points for sampling for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param sampling_spline_points The number of spline points to set.
     */
    void SetSamplingSplinePoints(mjModel* m, int sampling_spline_points) {
        int id_sampling_spline_points = mj_name2id(m, mjOBJ_NUMERIC, "sampling_spline_points");
        m->numeric_data[m->numeric_adr[id_sampling_spline_points]] = sampling_spline_points;
        RCLCPP_INFO_STREAM(this->get_logger(), "sampling_spline_points=" << sampling_spline_points);
    }

    /**
     * @brief Set the sampling representation for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param sampling_representation The sampling representation to set.
     */
    void SetSamplingRepresentation(mjModel* m, int sampling_representation) {
        int id_sampling_representation = mj_name2id(m, mjOBJ_NUMERIC, "sampling_representation");
        m->numeric_data[m->numeric_adr[id_sampling_representation]] = sampling_representation;
        RCLCPP_INFO_STREAM(this->get_logger(), "sampling_representation=" << sampling_representation);
    }

    /**
     * @brief Set the cost of position for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_pos The cost of position to set.
     */
    void SetCostPos(mjModel* m, double cost_pos) {
        int id_cost_pos                                   = mj_name2id(m, mjOBJ_SENSOR, "Cube Position");
        m->sensor_user[m->nuser_sensor * id_cost_pos + 1] = cost_pos;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_pos=" << cost_pos);
    }

    /**
     * @brief Set the cost of orientation for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_orientation The cost of orientation to set.
     */
    void SetCostOrientation(mjModel* m, double cost_orientation) {
        int id_cost_orientation                                   = mj_name2id(m, mjOBJ_SENSOR, "Cube Orientation");
        m->sensor_user[m->nuser_sensor * id_cost_orientation + 1] = cost_orientation;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_orientation=" << cost_orientation);
    }

    /**
     * @brief Set the cost of actuation for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_actuation The cost of actuation to set.
     */
    void SetCostActuation(mjModel* m, double cost_actuation) {
        int id_cost_actuation                                   = mj_name2id(m, mjOBJ_SENSOR, "Actuation");
        m->sensor_user[m->nuser_sensor * id_cost_actuation + 1] = cost_actuation;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_actuation=" << cost_actuation);
    }

    /**
     * @brief Set the cost of grasp for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_grasp The cost of grasp to set.
     */
    void SetCostGrasp(mjModel* m, double cost_grasp) {
        int id_cost_grasp                                   = mj_name2id(m, mjOBJ_SENSOR, "Grasp");
        m->sensor_user[m->nuser_sensor * id_cost_grasp + 1] = cost_grasp;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_grasp=" << cost_grasp);
    }

    /**
     * @brief Set the cost of linear velocity for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_cube_linear_velocity The cost of linear velocity to set.
     */
    void SetCostCubeLinearVelocity(mjModel* m, double cost_cube_linear_velocity) {
        int id_cost_cube_linear_velocity = mj_name2id(m, mjOBJ_SENSOR, "Cube Velocity");
        m->sensor_user[m->nuser_sensor * id_cost_cube_linear_velocity + 1] = cost_cube_linear_velocity;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_cube_linear_velocity=" << cost_cube_linear_velocity);
    }

    /**
     * @brief Set the cost of angular velocity for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param cost_cube_angular_velocity The cost of angular velocity to set.
     */
    void SetCostCubeAngularVelocity(mjModel* m, double cost_cube_angular_velocity) {
        int id_cost_cube_angular_velocity = mj_name2id(m, mjOBJ_SENSOR, "Cube Angular Velocity");
        m->sensor_user[m->nuser_sensor * id_cost_cube_angular_velocity + 1] = cost_cube_angular_velocity;
        RCLCPP_INFO_STREAM(this->get_logger(), "cost_cube_angular_velocity=" << cost_cube_angular_velocity);
    }

    /**
     * @brief Set the minimum standard deviation for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param std_min The minimum standard deviation to set.
     */
    void SetMinStd(mjModel* m, double std_min) {
        int id_std_min                              = mj_name2id(m, mjOBJ_NUMERIC, "std_min");
        m->numeric_data[m->numeric_adr[id_std_min]] = std_min;
        RCLCPP_INFO_STREAM(this->get_logger(), "std_min=" << std_min);
    }

    /**
     * @brief Set the sampling exploration for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param sampling_exploration The sampling exploration to set.
     */
    void SetSamplingExploration(mjModel* m, double sampling_exploration) {
        int id_sampling_exploration                              = mj_name2id(m, mjOBJ_NUMERIC, "sampling_exploration");
        m->numeric_data[m->numeric_adr[id_sampling_exploration]] = sampling_exploration;
        RCLCPP_INFO_STREAM(this->get_logger(), "sampling_exploration=" << sampling_exploration);
    }

    /**
     * @brief Set the axis-aligned goal for the given MJPC model.
     *
     * @param m Pointer to the MuJoCo model.
     * @param axis_aligned_goal The axis-aligned goal to set.
     */
    void SetAxisAlignedGoal(mjModel* m, int axis_aligned_goal) {
        int id_axis_aligned_goal                              = mj_name2id(m, mjOBJ_NUMERIC, "axis_aligned_goal");
        m->numeric_data[m->numeric_adr[id_axis_aligned_goal]] = axis_aligned_goal;
        RCLCPP_INFO_STREAM(this->get_logger(), "axis_aligned_goal=" << axis_aligned_goal);
    }
};

int main(int argc, char* argv[]) {
    obelisk::utils::SpinObelisk<CroController, rclcpp::executors::MultiThreadedExecutor>(argc, argv, "cro_controller");
}
