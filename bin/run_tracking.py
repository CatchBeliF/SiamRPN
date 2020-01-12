from lib.tracker import SiamRPNTracker
from got10k.experiments import *

if __name__ == '__main__':

    # tracking
    model_path = '/home/cbf/pycharmprojects/siamrpn/data/models/siamrpn_epoch_50.pth'
    tracker = SiamRPNTracker(model_path=model_path)

    experiments = [
        # ExperimentVOT('/home/cbf/datasets/vot2016', version=2016),
        ExperimentOTB('/home/cbf/datasets/OTB100/Benchmark', version=2015)
    ]

    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
