import os
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from JavaClones.BangBangTrajectory1D import BBTrajectoryPart
from JavaClones.Vec2 import Vec2
from Plotting.PlotType import PlotType
from Simulating.AlphaData import AlphaData
from Simulating.SimConfig import SimConfig
from Simulating.SimStep import SimStep1d, SimStep2d, SimStep
from Simulating.Simulator import Simulator

rcParams.update({'figure.autolayout': True})


@dataclass(frozen=True)
class Plotter:
    config: SimConfig
    save_fig: bool
    show_fig: bool

    def __post_init__(self):
        assert isinstance(self.save_fig, bool) or self.save_fig != "", "save_fig can not be empty file name"
        assert self.save_fig or self.show_fig, "Either save_fig or show_fig should be specified"

    def __getattr__(self, item):
        return getattr(self.config, item)

    @staticmethod
    def build_title(plot_type: PlotType, s: Union[float, Vec2], v0: Union[float, Vec2],
                    tt: float, custom_headings: List[str] = None) -> str:
        if custom_headings is None:
            custom_headings = list()
        s_str = "{:.3f}".format(s) if isinstance(s, float) else str(s)
        v0_str = "{:.3f}".format(v0) if isinstance(v0, float) else str(v0)
        tt_str = "{:.3f}\\,$s".format(tt) if tt is not None else "$None"

        return "{:s} $s = {:s}\\,$m | $v_0 = {:s}\\,$m/s | $t_t = {:s}" \
                   .format(plot_type.name, s_str, v0_str, tt_str) \
               + (" | " + " | ".join(custom_headings) if len(custom_headings) > 0 else "")

    def build_file_name(self, plot_type: PlotType) -> str:
        tt = "{:04.0f}".format(self.tt * 1000) if self.tt is not None else "none"
        if isinstance(self.s, float):
            return "{:s}_s{:04.0f}_v{:04.0f}_tt{:4s}_vM{:04.0f}_aM{:04.0f}" \
                .format(plot_type.name, self.s * 1000, self.v0 * 1000, tt, self.v_max * 1000, self.a_max * 1000)
        else:
            return "{:s}_s{:04.0f}x{:04.0f}_v{:04.0f}x{:04.0f}_tt{:4s}_vM{:04.0f}_aM{:04.0f}" \
                .format(plot_type.name, self.s.x * 1000, self.s.y * 1000, self.v0.x * 1000, self.v0.y * 1000, tt,
                        self.v_max * 1000, self.a_max * 1000)

    @staticmethod
    def plot(
            s0: Union[float, int, Vec2],
            s1: Union[float, int, Vec2],
            v0: Union[float, int, Vec2],
            v_max: Union[float, int],
            a_max: Union[float, int],
            tt: Union[None, float, int],
            primary_direction: Optional[Vec2],
            plot_type: PlotType,
            show_fig: bool = True,
            save_fig: Union[bool, str] = False,
            v_max_1d: Union[float, int, None] = None,
            a_max_1d: Union[float, int, None] = None,
    ):
        return Plotter(
            config=SimConfig(
                s=s1 - s0,
                v0=v0,
                v_max=v_max,
                a_max=a_max,
                v_max_1d=v_max_1d,
                a_max_1d=a_max_1d,
                tt=tt,
                primary_direction=primary_direction
            ),
            save_fig=save_fig,
            show_fig=show_fig
        )._plot(plot_type=plot_type)

    @staticmethod
    def plot_config(config: SimConfig, plot_type: PlotType, show_fig: bool = True, save_fig: Union[bool, str] = False):
        return Plotter(config=config, save_fig=save_fig, show_fig=show_fig)._plot(plot_type=plot_type)

    def _plot(self, plot_type: PlotType):
        match plot_type:
            case PlotType.TRAJ:
                return self._traj(draw_acc=True)
            case PlotType.TRAJ_NO_ACC:
                return self._traj(draw_acc=False)
            case PlotType.SIM_TRAJ:
                return self._sim_traj(draw_acc=True)
            case PlotType.SIM_TRAJ_NO_ACC:
                return self._sim_traj(draw_acc=False)
            case PlotType.DIFF_ALPHA:
                return self._diff_alpha()

    def _traj(self, draw_acc: bool):
        sim_steps = Simulator().simulate(self.config, 1, 300)
        fig = self._draw_last_sim_steps_from_list(sim_steps[:1], draw_acc=draw_acc)
        fig.suptitle(self.build_title(PlotType.TRAJ, self.s, self.v0, self.tt), fontsize=17)
        if isinstance(self.save_fig, str):
            fig.savefig(self.save_fig)
        elif self.save_fig:
            fig.savefig(self.build_file_name(PlotType.TRAJ) + ".svg")
        if not self.show_fig:
            plt.close(fig)
        return sim_steps[0].trajectory

    def _sim_traj(self, draw_acc: bool):
        sim_steps = Simulator().simulate(self.config, 30, 10)
        image_paths = []
        os.makedirs("./tmp", exist_ok=True)
        img_path = os.path.abspath(f"./tmp/{self.build_file_name(PlotType.SIM_TRAJ)}-")
        step_print = ""
        for i, step in enumerate(sim_steps):
            if isinstance(step, SimStep1d):
                step_print += f"({step.current_pos():.60f}, {step.current_vel():.60f}, {step.tt:.60f})  # {i}\r\n"
            elif isinstance(step, SimStep2d):
                step_print += f"(Vec2({step.current_pos().x:.60f},{step.current_pos().y:.60f}), Vec2(" \
                              f"{step.current_vel().x:.60f}, {step.current_vel().y:.60f}), {step.tt:.60f})  # {i}\r\n"
            fig = self._draw_last_sim_steps_from_list(sim_steps[:i + 1], draw_acc=draw_acc)
            title = self.build_title(PlotType.SIM_TRAJ, s=self.s - step.current_pos(),
                                     v0=step.current_vel(), tt=self.tt - step.current_time())
            title += " | {}".format(i)
            img_path_i = img_path + f"{i}.svg"
            fig.suptitle(title, fontsize=17)
            if self.save_fig:
                fig.savefig(img_path_i)
                image_paths.append(img_path_i)
            if not self.show_fig:
                plt.close(fig)

        if self.save_fig:
            gif_name = self.save_fig if isinstance(self.save_fig, str) else self.build_file_name(
                PlotType.SIM_TRAJ) + ".gif"
            print_name = self.build_file_name(PlotType.SIM_TRAJ) + ".txt"
            with imageio.get_writer(gif_name, mode="I") as writer:
                for path in image_paths:
                    image = imageio.v2.imread(path)
                    writer.append_data(image)
            with open(print_name, "w") as file:
                file.write(step_print)
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
        if isinstance(self.save_fig, str):
            fig.savefig(self.save_fig)
        elif self.save_fig:
            fig.savefig(self.build_file_name(PlotType.TRAJ) + ".svg")
        if not self.show_fig:
            plt.close(fig)

    def _fill_1d(self, ax_p, ax_v, ax_a, sim_steps: List[SimStep1d], s: float = None, axis_name: str = ""):
        times_passed = [sim_step.current_time() for sim_step in sim_steps]
        pos_passed = [sim_step.current_pos() for sim_step in sim_steps]
        vel_passed = [sim_step.current_vel() for sim_step in sim_steps]
        acc_passed = [sim_step.current_acc() for sim_step in sim_steps]

        ax_p.set_ylabel(f"Position {axis_name}[m]")
        ax_p.set_xlabel("time [s]")
        ax_p.plot(sim_steps[0].times, sim_steps[0].pos, color="gray")
        ax_p.scatter(times_passed, pos_passed, color="blue")
        ax_p.plot(times_passed, pos_passed, color="blue")
        ax_p.plot(sim_steps[-1].times, sim_steps[-1].pos, color="green")
        ax_p.grid(True)

        ax_v.set_ylim([-self.v_max - 0.25, self.v_max + 0.25])
        ax_v.set_ylabel(f"Velocity {axis_name}[m/s]")
        ax_v.set_xlabel("time [s]")
        ax_v.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].v_max, sim_steps[-1].v_max], color="black")
        ax_v.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].v_max, -sim_steps[-1].v_max], color="black")
        ax_v.plot(sim_steps[0].times, sim_steps[0].vel, color="gray")
        ax_v.scatter(times_passed, vel_passed, color="blue")
        ax_v.plot(times_passed, vel_passed, color="blue")
        ax_v.plot(sim_steps[-1].times, sim_steps[-1].vel, color="red")
        ax_v.grid(True)

        if ax_a is not None:
            ax_a.set_ylim([-self.a_max - 0.25, self.a_max + 0.25])
            ax_a.set_ylabel(f"Acceleration {axis_name}[m/s²]")
            ax_a.set_xlabel("time [s]")
            ax_a.plot([0, sim_steps[-1].times[-1]], [sim_steps[-1].a_max, sim_steps[-1].a_max], color="black")
            ax_a.plot([0, sim_steps[-1].times[-1]], [-sim_steps[-1].a_max, -sim_steps[-1].a_max], color="black")
            ax_a.plot(sim_steps[0].times, sim_steps[0].acc, color="gray")
            ax_a.scatter(times_passed, acc_passed, color="blue")
            ax_a.plot(times_passed, acc_passed, color="blue")
            ax_a.plot(sim_steps[-1].times, sim_steps[-1].acc)
            ax_a.grid(True)

        additional_ticks = []
        additional_tick_labels = []

        def mark_time(t: float, pos: Optional[float], color: str, name):
            if pos is not None:
                ax_p.scatter(x=t, y=pos, color=color, marker="x", s=20 * 6)
            else:
                lim = ax_p.get_ylim()
                ax_p.plot([t, t], [*lim], color=color)
                ax_p.set_ylim(lim)
            ax_v.plot([t, t], [-self.v_max - 0.25, self.v_max + 0.25], color=color)
            if ax_a is not None:
                ax_a.plot([t, t], [-self.a_max - 0.25, self.a_max + 0.25], color=color)
            additional_ticks.append(t + 0.0001)
            additional_tick_labels.append(name)

        if self.tt is not None and s is not None:
            mark_time(self.tt, s, "red", "\n$t_t$")

        mark_time(0, None, "gray", "\n$t_0$")
        for i in range(sim_steps[-1].trajectory.numParts):
            part = sim_steps[-1].trajectory.parts[i]
            if np.isclose(part.t_end + 0.0001, additional_ticks[-1]):
                additional_tick_labels[-1] += f"$,t_{i+1}$"
            else:
                mark_time(part.t_end, None, "gray", f"\n$t_{i + 1}$")

        ax_p.set_xticks(additional_ticks, additional_tick_labels, minor=True)
        ax_v.set_xticks(additional_ticks, additional_tick_labels, minor=True)
        if ax_a is not None:
            ax_a.set_xticks(additional_ticks, additional_tick_labels, minor=True, va="top")

    def _fill_2d(self, ax_3d, ax_alpha, sim_steps: List[SimStep2d]):
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

        if ax_alpha is not None:
            ax_alpha.set_xlabel("alpha [rad]")
            ax_alpha.set_ylabel("time [s]")
            ax_alpha.set_ylim([-1, 5])
            ax_alpha.set_xlim([-0.05, np.pi * 0.5 + 0.05])
            self._fill_alpha(ax_alpha, sim_steps[-1].alpha_data)
            ax_alpha.plot([sim_steps[-1].alpha, sim_steps[-1].alpha], [-5, 5], color="red", label="chosen", ls="--")
            ax_alpha.legend()
            ax_alpha.plot()

        if self.tt is not None:
            ax_3d.scatter(self.s.x, self.s.y, self.tt, color="red", marker="x", s=20 * 6)

    @staticmethod
    def _fill_alpha(ax_alpha, data: AlphaData):
        ax_alpha.plot([-5, 5], [0, 0], color="gray")
        ax_alpha.plot(data.alphas, data.diffs, label="|x-y|", color="orange")
        ax_alpha.plot(data.alphas, data.x_times, label="x", color="blue")
        ax_alpha.plot(data.alphas, data.y_times, label="y", color="cyan")
        ax_alpha.plot([data.optimal, data.optimal], [-5, 5], color="green", label="optimal")

    def _draw_last_sim_steps_from_list(self, sim_steps: List[SimStep], draw_acc: bool) -> plt.Figure:
        if isinstance(sim_steps[0], SimStep1d):
            # 1D Trajectory
            sim_steps_1d = [step for step in sim_steps if isinstance(step, SimStep1d)]
            if draw_acc:
                fig, (ax_p, ax_v, ax_a) = plt.subplots(1, 3, figsize=(12, 3.5))
            else:
                fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(8, 3.5))
                ax_a = None
            self._fill_1d(ax_p, ax_v, ax_a, s=self.s, sim_steps=sim_steps_1d)

        elif isinstance(sim_steps[0], SimStep2d):
            # 2D Trajectory
            sim_steps_2d = [step for step in sim_steps if isinstance(step, SimStep2d)]
            sim_steps_x = [step.get_1d_x() for step in sim_steps_2d]
            sim_steps_y = [step.get_1d_y() for step in sim_steps_2d]

            if draw_acc:
                fig = plt.figure(figsize=(12, 10.5))
                ax_3d = fig.add_subplot(3, 3, (1, 2), projection="3d")
                ax_alpha = fig.add_subplot(333)
                self._fill_2d(ax_3d, ax_alpha, sim_steps=sim_steps_2d)
                ax_p = fig.add_subplot(334)
                ax_v = fig.add_subplot(335)
                ax_a = fig.add_subplot(336)
                self._fill_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_x, s=self.s.x, axis_name="x ")
                ax_p = fig.add_subplot(337)
                ax_v = fig.add_subplot(338)
                ax_a = fig.add_subplot(339)
                self._fill_1d(ax_p, ax_v, ax_a, sim_steps=sim_steps_y, s=self.s.y, axis_name="y ")
            else:
                fig = plt.figure(figsize=(12, 10.5))
                ax_3d = fig.add_subplot(3, 2, (1, 2), projection="3d")
                self._fill_2d(ax_3d, None, sim_steps=sim_steps_2d)
                ax_p = fig.add_subplot(323)
                ax_v = fig.add_subplot(324)
                self._fill_1d(ax_p, ax_v, None, sim_steps=sim_steps_x, s=self.s.x, axis_name="x ")
                ax_p = fig.add_subplot(325)
                ax_v = fig.add_subplot(326)
                self._fill_1d(ax_p, ax_v, None, sim_steps=sim_steps_y, s=self.s.y, axis_name="y ")
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
