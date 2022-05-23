import dataclasses
import math
import os
import traceback
from enum import IntEnum
from typing import List, Optional, Union, Tuple

import imageio
import numpy as np
from matplotlib import pyplot as plt

from simulating import Simulator, SimStep1d, SimStep2d, SimStep
from traj import Vec2
from traj1D import BBTrajectoryPart, BangBangTrajectory1D
from traj2D import BangBangTrajectory2D


class PlotType(IntEnum):
    NONE = 0
    TRAJ = 1
    SIM_TRAJ = 2
    SLOWEST_DIRECT = 3
    FASTEST_DIRECT = 4
    FASTEST_OVERSHOT = 5
    CAN_REACH = 6
    DIFF_ALPHA = 7


@dataclasses.dataclass(frozen=True)
class Plotter:
    distance: Union[float, Vec2]
    initial_vel: Union[float, Vec2]
    target_time: Optional[float]
    max_vel: float
    max_acc: float
    save_fig: bool
    show_fig: bool

    def __post_init__(self):
        assert isinstance(self.distance, type(self.initial_vel))

    @staticmethod
    def build_title(plot_type: PlotType, distance: Union[float, Vec2], initial_vel: Union[float, Vec2],
                    target_time: float, custom_headings: List[str] = None) -> str:
        if custom_headings is None:
            custom_headings = list()
        distance_str = "{:.3f}".format(distance) if isinstance(distance, float) else str(distance)
        initial_vel_str = "{:.3f}".format(initial_vel) if isinstance(initial_vel, float) else str(initial_vel)
        target_time_str = "{:.3f}".format(target_time) if target_time is not None else "None"

        return "{:s} d = {:s} m | v0 = {:s} m/s | tt = {:s} s" \
                   .format(plot_type.name, distance_str, initial_vel_str, target_time_str) \
               + (" | " + " | ".join(custom_headings) if len(custom_headings) > 0 else "")

    def build_file_name(self, plot_type: PlotType) -> str:
        distance = self.distance if isinstance(self.distance, float) else self.distance.length()
        initial_vel = self.initial_vel if isinstance(self.initial_vel, float) else self.initial_vel.length()
        target_time = "{:04.0f}".format(self.target_time * 1000) if self.target_time is not None else "none"

        return "{:s}_d{:04.0f}_v{:04.0f}_tt{:4s}_vM{:04.0f}_aM{:04.0f}" \
            .format(plot_type.name, distance * 1000, initial_vel * 1000, target_time, self.max_vel * 1000,
                    self.max_acc * 1000)

    @staticmethod
    def plot(distance: Union[float, Vec2, Tuple[float, float]], initial_vel: Union[float, Vec2, Tuple[float, float]],
             target_time: Optional[float], max_vel: float, max_acc: float, plot_type: PlotType,
             plot_fallback: PlotType = PlotType.NONE, show_fig: bool = True, save_fig: bool = False):
        return Plotter(distance=Vec2(*distance) if isinstance(distance, tuple) else distance,
                       initial_vel=Vec2(*initial_vel) if isinstance(initial_vel, tuple) else initial_vel,
                       target_time=target_time,
                       max_vel=max_vel,
                       max_acc=max_acc,
                       save_fig=save_fig,
                       show_fig=show_fig) \
            ._plot(plot_type=plot_type, plot_fallback=plot_fallback)

    def _plot(self, plot_type: PlotType, plot_fallback: PlotType = None):
        try:
            match plot_type:
                case PlotType.TRAJ:
                    return self._traj()
                case PlotType.SIM_TRAJ:
                    return self._sim_traj()
                case PlotType.SLOWEST_DIRECT:
                    return self._slowest_direct()
                case PlotType.FASTEST_DIRECT:
                    return self._fastest_direct()
                case PlotType.FASTEST_OVERSHOT:
                    return self._fastest_overshot()
                case PlotType.CAN_REACH:
                    return self._can_reach()
                case PlotType.DIFF_ALPHA:
                    return self._diff_alpha()

        except Exception as e:
            print("{}, {}, {}, {} failed with:"
                  .format(self.distance, self.initial_vel, self.target_time, plot_type.name))
            print(type(e))
            print(e)
            print(traceback.format_exc())
            if plot_fallback is not None:
                return self._plot(plot_fallback)
            else:
                raise AssertionError

    def _traj(self):
        sim_steps = Simulator(self.max_vel, self.max_acc, num_steps=1, step_size=300, distance=self.distance,
                              initial_vel=self.initial_vel, target_time=self.target_time).simulate()
        fig = self._draw_last_sim_steps_from_list(sim_steps[:1])
        fig.suptitle(self.build_title(PlotType.TRAJ, self.distance, self.initial_vel, self.target_time))
        if self.save_fig:
            fig.savefig(self.build_file_name(PlotType.TRAJ) + ".png")
        if not self.show_fig:
            plt.close(fig)
        return sim_steps[0].trajectory

    def _sim_traj(self):
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        image_paths = []

        sim_steps = Simulator(self.max_vel, self.max_acc, num_steps=30, step_size=10, distance=self.distance,
                              initial_vel=self.initial_vel, target_time=self.target_time).simulate_real()

        for i, step in enumerate(sim_steps):
            fig = self._draw_last_sim_steps_from_list(sim_steps[:i + 1])
            title = self.build_title(PlotType.SIM_TRAJ, distance=self.distance - step.current_pos(),
                                     initial_vel=step.current_vel(), target_time=self.target_time - step.current_time())
            title += " | {}".format(i)
            # print("{}: ({}), ({}), {}".format(i, self.distance - sim_steps[i].current_pos(),
            # sim_steps[i].current_vel(), self.target_time - sim_steps[i].current_time()))
            img_path = os.path.abspath("./tmp/{}.png".format(i))
            fig.suptitle(title, fontsize=20)
            fig.savefig(img_path)
            if not self.show_fig:
                plt.close(fig)
            image_paths.append(img_path)

        gif_name = self.build_file_name(PlotType.SIM_TRAJ) + ".gif"
        with imageio.get_writer(gif_name, mode="I") as writer:
            for path in image_paths:
                image = imageio.v2.imread(path)
                writer.append_data(image)
        for path in set(image_paths):
            os.remove(path)

    def _slowest_direct(self):
        if isinstance(self.distance, Vec2):
            raise NotImplementedError
        part = BangBangTrajectory1D.calc_slowest_direct(0, self.distance, self.initial_vel, self.max_acc)
        if part is None:
            self._traj()
        data = self._get_data_from_parts(part)
        assert math.isclose(data[1][-1], self.distance, abs_tol=1e-6), "{} != {}".format(data[1][-1], self.distance)
        self._plot_data_(*data, plot_type=PlotType.SLOWEST_DIRECT)

    def _fastest_direct(self):
        if isinstance(self.distance, Vec2):
            raise NotImplementedError
        parts = BangBangTrajectory1D.calc_fastest_direct(0, self.distance, self.initial_vel, self.max_vel, self.max_acc)
        data = self._get_data_from_parts(parts)
        assert math.isclose(data[1][-1], self.distance, abs_tol=1e-6), "{} != {}".format(data[1][-1], self.distance)
        self._plot_data_(*data, plot_type=PlotType.FASTEST_DIRECT)

    def _fastest_overshot(self):
        if isinstance(self.distance, Vec2):
            raise NotImplementedError
        part = BangBangTrajectory1D.calc_slowest_direct(0, self.distance, self.initial_vel, self.max_acc)
        if part is None:
            return
        self._slowest_direct()
        parts = BangBangTrajectory1D.calc_fastest_overshot(0, self.distance, self.initial_vel, self.max_vel,
                                                           self.max_acc)
        data = self._get_data_from_parts(parts)
        assert math.isclose(data[1][-1], self.distance, abs_tol=1e-6), "{} != {}".format(data[1][-1], self.distance)
        self._plot_data_(*data, plot_type=PlotType.FASTEST_OVERSHOT)

    def _can_reach(self):
        if isinstance(self.distance, Vec2):
            raise NotImplementedError
        can_reach, parts, reason, time_remaining = BangBangTrajectory1D.can_reach(
            0, self.distance, self.initial_vel, self.max_vel, self.max_acc, self.target_time)
        data = self._get_data_from_parts(parts)
        self._plot_data_(*data, plot_type=PlotType.CAN_REACH,
                         custom_headings=["can = {}".format(can_reach), reason, "tr = {:.3f}s".format(time_remaining)])

    def _diff_alpha(self):
        if isinstance(self.distance, (int, float)):
            raise NotImplementedError

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        fig.suptitle(self.build_title(PlotType.DIFF_ALPHA, self.distance, self.initial_vel, self.target_time))
        self._fill_alpha(ax, self.distance, self.initial_vel, self.max_vel, self.max_acc, self.target_time)
        ax.legend()
        ax.set_ylim([-5, 5])
        if self.save_fig:
            fig.savefig(self.build_file_name(PlotType.DIFF_ALPHA))
        if not self.show_fig:
            plt.close(fig)

    def _fill_static_1d(self, ax_p, ax_v, ax_a,
                        times: Union[np.ndarray, List[float]],
                        pos: Union[np.ndarray, List[float]],
                        vel: Union[np.ndarray, List[float]],
                        acc: Union[np.ndarray, List[float]]):

        ax_p.set_ylabel("Position [m]")
        ax_p.set_xlabel("time [s]")
        ax_p.plot(times, pos, color="green")
        ax_p.grid(True)

        ax_v.set_ylim([-self.max_vel - 0.25, self.max_vel + 0.25])
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_xlabel("time [s]")
        ax_v.plot(times, vel, color="red")
        ax_v.grid(True)

        ax_a.set_ylim([-self.max_acc - 0.25, self.max_acc + 0.25])
        ax_a.set_ylabel("Acceleration [m/s²]")
        ax_a.set_xlabel("time [s]")
        ax_a.plot(times, acc)
        ax_a.grid(True)

        if self.target_time is not None:
            ax_p.scatter(x=self.target_time, y=self.distance, color="red", marker="x", s=20 * 6)
            ax_v.plot([self.target_time, self.target_time], [-self.max_vel - 0.25, self.max_vel + 0.25], color="gray")
            ax_a.plot([self.target_time, self.target_time], [-self.max_acc - 0.25, self.max_acc + 0.25], color="gray")

    def _fill_dynamic_1d(self, ax_p, ax_v, ax_a, sim_steps: List[SimStep1d], distance: float = None):
        times_passed = [sim_step.current_time() for sim_step in sim_steps]
        pos_passed = [sim_step.current_pos() for sim_step in sim_steps]
        vel_passed = [sim_step.current_vel() for sim_step in sim_steps]
        acc_passed = [sim_step.current_acc() for sim_step in sim_steps]

        ax_p.set_ylabel("Position [m]")
        ax_p.set_xlabel("time [s]")
        ax_p.plot(sim_steps[0].times, sim_steps[0].pos, color="gray")
        ax_p.scatter(times_passed, pos_passed, color="blue")
        ax_p.plot(times_passed, pos_passed, color="blue")
        ax_p.plot(sim_steps[-1].times, sim_steps[-1].pos, color="green")
        ax_p.grid(True)

        ax_v.set_ylim([-self.max_vel - 0.25, self.max_vel + 0.25])
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_xlabel("time [s]")
        ax_v.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].max_vel, sim_steps[-1].max_vel], color="black")
        ax_v.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].max_vel, -sim_steps[-1].max_vel], color="black")
        ax_v.plot(sim_steps[0].times, sim_steps[0].vel, color="gray")
        ax_v.scatter(times_passed, vel_passed, color="blue")
        ax_v.plot(times_passed, vel_passed, color="blue")
        ax_v.plot(sim_steps[-1].times, sim_steps[-1].vel, color="red")
        ax_v.grid(True)

        ax_a.set_ylim([-self.max_acc - 0.25, self.max_acc + 0.25])
        ax_a.set_ylabel("Acceleration [m/s²]")
        ax_a.set_xlabel("time [s]")
        ax_a.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].max_acc, sim_steps[-1].max_acc], color="black")
        ax_a.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].max_acc, -sim_steps[-1].max_acc], color="black")
        ax_a.plot(sim_steps[0].times, sim_steps[0].acc, color="gray")
        ax_a.scatter(times_passed, acc_passed, color="blue")
        ax_a.plot(times_passed, acc_passed, color="blue")
        ax_a.plot(sim_steps[-1].times, sim_steps[-1].acc)
        ax_a.grid(True)

        if self.target_time is not None and distance is not None:
            ax_p.scatter(x=self.target_time, y=distance, color="red", marker="x", s=20 * 6)
            ax_v.plot([self.target_time, self.target_time], [-self.max_vel - 0.25, self.max_vel + 0.25], color="gray")
            ax_a.plot([self.target_time, self.target_time], [-self.max_acc - 0.25, self.max_acc + 0.25], color="gray")

    def _fill_dynamic_2d(self, ax_3d, ax_alpha, sim_steps: List[SimStep2d]):
        passed_times = [sim_step.current_time() for sim_step in sim_steps]
        passed_x = [sim_step.current_pos().x for sim_step in sim_steps]
        passed_y = [sim_step.current_pos().y for sim_step in sim_steps]

        total_x = [pos.x for pos in sim_steps[0].pos]
        total_y = [pos.y for pos in sim_steps[0].pos]

        future_x = [pos.x for pos in sim_steps[-1].pos]
        future_y = [pos.y for pos in sim_steps[-1].pos]

        ax_3d.set_xlabel("Position x [m]")
        ax_3d.set_ylabel("Position y [m]")
        ax_3d.set_zlabel("time [s]")
        ax_3d.plot(total_x, total_y, sim_steps[0].times, color="gray")
        ax_3d.scatter(passed_x, passed_y, passed_times, color="blue")
        ax_3d.plot(passed_x, passed_y, passed_times, color="blue")
        ax_3d.plot(future_x, future_y, sim_steps[-1].times, color="green")
        ax_3d.grid(True)

        ax_alpha.set_xlabel("Alpha [rad]")
        ax_alpha.set_ylabel("time [s]")
        ax_alpha.set_ylim([-5, 5])
        ax_alpha.plot([sim_steps[-1].alpha, sim_steps[-1].alpha], [-5, 5], color="red", label="chosen")
        ax_alpha.plot([sim_steps[-1].optimal_alpha, sim_steps[-1].optimal_alpha], [-5, 5], color="green",
                      label="optimal")

        self._fill_alpha(ax_alpha, sim_steps[-1].pos[-1] - sim_steps[-1].current_pos(), sim_steps[-1].current_vel(),
                         self.max_vel, self.max_acc,
                         self.target_time - sim_steps[-1].current_time() if self.target_time is not None else None)
        ax_alpha.legend()
        ax_alpha.plot()

        if self.target_time is not None:
            ax_3d.scatter(self.distance.x, self.distance.y, self.target_time, color="red", marker="x", s=20 * 6)

    @staticmethod
    def _fill_alpha(ax_alpha, distance, initial_vel, max_vel, max_acc, target_time):

        alphas = np.linspace(1e-4, math.pi * 0.5 - 1e-4, num=500)
        data = [
            BangBangTrajectory2D.diff_for_alpha(a, Vec2(0, 0), distance, initial_vel, max_vel, max_acc, target_time) for
            a in alphas]

        diff = [e[0] for e in data]
        x = [e[1] for e in data]
        y = [e[2] for e in data]

        ax_alpha.plot(alphas, diff, label="diff", color="orange")
        ax_alpha.plot(alphas, x, label="x", color="blue")
        ax_alpha.plot(alphas, y, label="y", color="cyan")

    def _draw_last_sim_steps_from_list(self, sim_steps: List[SimStep]) -> plt.Figure:
        if isinstance(sim_steps[0], SimStep1d):
            # 1D Trajectory
            sim_steps_1d = [step for step in sim_steps if isinstance(step, SimStep1d)]
            fig, (ax_p, ax_v, ax_a) = plt.subplots(1, 3, figsize=(20, 5))
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, distance=self.distance, sim_steps=sim_steps_1d)

        elif isinstance(sim_steps[0], SimStep2d):
            # 2D Trajectory
            sim_steps_2d = [step for step in sim_steps if isinstance(step, SimStep2d)]
            sim_steps_x = [step.get_1d_x() for step in sim_steps_2d]
            sim_steps_y = [step.get_1d_y() for step in sim_steps_2d]

            fig = plt.figure(figsize=(20, 15))
            ax_3d = fig.add_subplot(3, 3, (1, 2), projection="3d")
            ax_alpha = fig.add_subplot(333)
            self._fill_dynamic_2d(ax_3d, ax_alpha, sim_steps=sim_steps_2d)
            ax_p = fig.add_subplot(334)
            ax_v = fig.add_subplot(335)
            ax_a = fig.add_subplot(336)
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_x, distance=self.distance.x)
            ax_p = fig.add_subplot(337)
            ax_v = fig.add_subplot(338)
            ax_a = fig.add_subplot(339)
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_y, distance=self.distance.y)
            pass
        else:
            raise ValueError
        return fig

    @staticmethod
    def _get_data_from_part(part: BBTrajectoryPart, t_start=0.0) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        times = np.linspace(t_start, part.t_end, 200)
        acc = np.ones(times.shape) * part.acc
        vel = part.v0 + acc * (times - t_start)
        pos = part.s0 + 0.5 * (part.v0 + vel) * (times - t_start)
        return times, pos, vel, acc

    @staticmethod
    def _get_data_from_parts(parts: List[BBTrajectoryPart], t_start=0.0) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        times, acc, vel, pos = None, None, None, None
        t_start_new = t_start
        for part in parts:
            times_new, pos_new, vel_new, acc_new = Plotter._get_data_from_part(part, t_start=t_start_new)
            t_start_new = part.t_end
            times = np.append(times, times_new) if times is not None else times_new
            pos = np.append(pos, pos_new) if pos is not None else pos_new
            vel = np.append(vel, vel_new) if vel is not None else vel_new
            acc = np.append(acc, acc_new) if acc is not None else acc_new
        return times, pos, vel, acc

    def _plot_data_(self, times, pos, vel, acc, plot_type: PlotType, custom_headings: List[str] = None):
        fig, (ax_p, ax_v, ax_a) = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(self.build_title(plot_type, self.distance, self.initial_vel, self.target_time, custom_headings))
        self._fill_static_1d(ax_p, ax_v, ax_a, times, pos, vel, acc)
        if self.save_fig:
            fig.savefig(self.build_file_name(plot_type))
        if not self.show_fig:
            plt.close(fig)
