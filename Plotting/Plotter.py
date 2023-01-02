import os
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np

from JavaClones.BangBangTrajectory1D import BBTrajectoryPart
from JavaClones.Vec2 import Vec2
from Plotting.PlotType import PlotType
from Simulating.AlphaData import AlphaData
from Simulating.SimConfig import SimConfig
from Simulating.SimStep import SimStep1d, SimStep2d, SimStep
from Simulating.Simulator import Simulator


@dataclass(frozen=True)
class Plotter:
    config: SimConfig
    save_fig: bool
    show_fig: bool

    def __getattr__(self, item):
        return getattr(self.config, item)

    @staticmethod
    def build_title(plot_type: PlotType, s: Union[float, Vec2], v0: Union[float, Vec2],
                    tt: float, custom_headings: List[str] = None) -> str:
        if custom_headings is None:
            custom_headings = list()
        s_str = "{:.3f}".format(s) if isinstance(s, float) else str(s)
        v0_str = "{:.3f}".format(v0) if isinstance(v0, float) else str(v0)
        tt_str = "{:.3f}".format(tt) if tt is not None else "None"

        return "{:s} d = {:s} m | v0 = {:s} m/s | tt = {:s} s" \
                   .format(plot_type.name, s_str, v0_str, tt_str) \
               + (" | " + " | ".join(custom_headings) if len(custom_headings) > 0 else "")

    def build_file_name(self, plot_type: PlotType) -> str:
        tt = "{:04.0f}".format(self.tt * 1000) if self.tt is not None else "none"
        if isinstance(self.s, float):
            return "{:s}_d{:04.0f}_v{:04.0f}_tt{:4s}_vM{:04.0f}_aM{:04.0f}" \
                .format(plot_type.name, self.s * 1000, self.v0 * 1000, tt, self.v_max * 1000, self.a_max * 1000)
        else:
            return "{:s}_d{:04.0f}x{:04.0f}_v{:04.0f}x{:04.0f}_tt{:4s}_vM{:04.0f}_aM{:04.0f}" \
                .format(plot_type.name, self.s.x * 1000, self.s.y * 1000, self.v0.x * 1000, self.v0.y * 1000, tt,
                        self.v_max * 1000, self.a_max * 1000)

    @staticmethod
    def plot(
            s0: Union[float, Vec2],
            s1: Union[float, Vec2],
            v0: Union[float, Vec2],
            v_max: float,
            a_max: float,
            tt: Optional[float],
            primary_direction: Union[float, Vec2, None],
            plot_type: PlotType,
            show_fig: bool = True,
            save_fig: bool = False,
    ):

        return Plotter(
            config=SimConfig(
                s=s1 - s0,
                v0=v0,
                v_max=v_max,
                a_max=a_max,
                tt=tt,
                primary_direction=primary_direction
            ),
            save_fig=save_fig,
            show_fig=show_fig
        )._plot(plot_type=plot_type)

    @staticmethod
    def plot_config(config: SimConfig, plot_type: PlotType, show_fig: bool = True, save_fig: bool = False):
        return Plotter(config=config, save_fig=save_fig, show_fig=show_fig)._plot(plot_type=plot_type)

    def _plot(self, plot_type: PlotType):
        match plot_type:
            case PlotType.TRAJ:
                return self._traj()
            case PlotType.SIM_TRAJ:
                return self._sim_traj()
            case PlotType.DIFF_ALPHA:
                return self._diff_alpha()

    def _traj(self):
        sim_steps = Simulator().simulate(self.config, 1, 300)
        fig = self._draw_last_sim_steps_from_list(sim_steps[:1])
        fig.suptitle(self.build_title(PlotType.TRAJ, self.s, self.v0, self.tt))
        if self.save_fig:
            fig.savefig(self.build_file_name(PlotType.TRAJ) + ".png")
        if not self.show_fig:
            plt.close(fig)
        return sim_steps[0].trajectory

    def _sim_traj(self):
        sim_steps = Simulator().simulate(self.config, 300, 1)
        image_paths = []
        os.makedirs("./tmp", exist_ok=True)
        img_path = os.path.abspath(f"./tmp/{self.build_file_name(PlotType.SIM_TRAJ)}-")
        for i, step in enumerate(sim_steps):
            fig = self._draw_last_sim_steps_from_list(sim_steps[:i + 1])
            title = self.build_title(PlotType.SIM_TRAJ, s=self.s - step.current_pos(),
                                     v0=step.current_vel(), tt=self.tt - step.current_time())
            title += " | {}".format(i)
            # print("{}: ({}), ({}), {}".format(i, self.s - sim_steps[i].current_pos(),
            # sim_steps[i].current_vel(), self.tt - sim_steps[i].current_time()))
            img_path_i = img_path + f"{i}.png"
            fig.suptitle(title, fontsize=20)
            fig.savefig(img_path_i)
            if not self.show_fig:
                plt.close(fig)
            image_paths.append(img_path_i)

        gif_name = self.build_file_name(PlotType.SIM_TRAJ) + ".gif"
        with imageio.get_writer(gif_name, mode="I") as writer:
            for path in image_paths:
                image = imageio.v2.imread(path)
                writer.append_data(image)
        for path in set(image_paths):
            os.remove(path)
        return sim_steps[0].trajectory

    def _diff_alpha(self):
        if isinstance(self.s, (int, float)):
            raise NotImplementedError

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        fig.suptitle(self.build_title(PlotType.DIFF_ALPHA, self.s, self.v0, self.tt))
        data = Simulator.create_alpha_data(self.config)
        self._fill_alpha(ax, data)
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

        ax_v.set_ylim([-self.v_max - 0.25, self.v_max + 0.25])
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_xlabel("time [s]")
        ax_v.plot(times, vel, color="red")
        ax_v.grid(True)

        ax_a.set_ylim([-self.a_max - 0.25, self.a_max + 0.25])
        ax_a.set_ylabel("Acceleration [m/s²]")
        ax_a.set_xlabel("time [s]")
        ax_a.plot(times, acc)
        ax_a.grid(True)

        if self.tt is not None:
            ax_p.scatter(x=self.tt, y=self.s, color="red", marker="x", s=20 * 6)
            ax_v.plot([self.tt, self.tt], [-self.v_max - 0.25, self.v_max + 0.25], color="gray")
            ax_a.plot([self.tt, self.tt], [-self.a_max - 0.25, self.a_max + 0.25], color="gray")

    def _fill_dynamic_1d(self, ax_p, ax_v, ax_a, sim_steps: List[SimStep1d], s: float = None):
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

        ax_v.set_ylim([-self.v_max - 0.25, self.v_max + 0.25])
        ax_v.set_ylabel("Velocity [m/s]")
        ax_v.set_xlabel("time [s]")
        ax_v.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].v_max, sim_steps[-1].v_max], color="black")
        ax_v.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].v_max, -sim_steps[-1].v_max], color="black")
        ax_v.plot(sim_steps[0].times, sim_steps[0].vel, color="gray")
        ax_v.scatter(times_passed, vel_passed, color="blue")
        ax_v.plot(times_passed, vel_passed, color="blue")
        ax_v.plot(sim_steps[-1].times, sim_steps[-1].vel, color="red")
        ax_v.grid(True)

        ax_a.set_ylim([-self.a_max - 0.25, self.a_max + 0.25])
        ax_a.set_ylabel("Acceleration [m/s²]")
        ax_a.set_xlabel("time [s]")
        ax_a.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].a_max, sim_steps[-1].a_max], color="black")
        ax_a.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].a_max, -sim_steps[-1].a_max], color="black")
        ax_a.plot(sim_steps[0].times, sim_steps[0].acc, color="gray")
        ax_a.scatter(times_passed, acc_passed, color="blue")
        ax_a.plot(times_passed, acc_passed, color="blue")
        ax_a.plot(sim_steps[-1].times, sim_steps[-1].acc)
        ax_a.grid(True)

        if self.tt is not None and s is not None:
            ax_p.scatter(x=self.tt, y=s, color="red", marker="x", s=20 * 6)
            ax_v.plot([self.tt, self.tt], [-self.v_max - 0.25, self.v_max + 0.25], color="gray")
            ax_a.plot([self.tt, self.tt], [-self.a_max - 0.25, self.a_max + 0.25], color="gray")

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
        self._fill_alpha(ax_alpha, sim_steps[-1].alpha_data)
        ax_alpha.plot([sim_steps[-1].alpha, sim_steps[-1].alpha], [-5, 5], color="red", label="chosen", ls="--")
        ax_alpha.legend()
        ax_alpha.plot()

        if self.tt is not None:
            ax_3d.scatter(self.s.x, self.s.y, self.tt, color="red", marker="x", s=20 * 6)

    @staticmethod
    def _fill_alpha(ax_alpha, data: AlphaData):
        ax_alpha.plot(data.alphas, data.diffs, label="diff", color="orange")
        ax_alpha.plot(data.alphas, data.x_times, label="x", color="blue")
        ax_alpha.plot(data.alphas, data.y_times, label="y", color="cyan")
        ax_alpha.plot([data.optimal, data.optimal], [-5, 5], color="green", label="optimal")

    def _draw_last_sim_steps_from_list(self, sim_steps: List[SimStep]) -> plt.Figure:
        if isinstance(sim_steps[0], SimStep1d):
            # 1D Trajectory
            sim_steps_1d = [step for step in sim_steps if isinstance(step, SimStep1d)]
            fig, (ax_p, ax_v, ax_a) = plt.subplots(1, 3, figsize=(20, 5))
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, s=self.s, sim_steps=sim_steps_1d)

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
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_x, s=self.s.x)
            ax_p = fig.add_subplot(337)
            ax_v = fig.add_subplot(338)
            ax_a = fig.add_subplot(339)
            self._fill_dynamic_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_y, s=self.s.y)
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
        fig.suptitle(self.build_title(plot_type, self.s, self.v0, self.tt, custom_headings))
        self._fill_static_1d(ax_p, ax_v, ax_a, times, pos, vel, acc)
        if self.save_fig:
            fig.savefig(self.build_file_name(plot_type))
        if not self.show_fig:
            plt.close(fig)
