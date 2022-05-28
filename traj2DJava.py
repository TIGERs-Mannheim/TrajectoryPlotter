from jnius import autoclass

if __name__ == '__main__':
    BangBangTrajectory1D = autoclass(
        ".home.michael.TIGERs.SumatraGnu.modules.common.src.main.java.edu.tigers.sumatra.trajectory.BangBangTrajectory1D".replace(
            "/", "."))
