import time

from .trainer import Trainer


def train_headless(trainer: Trainer, minutes, save_path=None, report_every=10.0, save_every=60.0):
    end = time.time() + minutes * 60
    last_report = last_save = time.time()
    while time.time() < end:
        trainer.run(0.5)
        now = time.time()
        if now - last_report >= report_every:
            last_report = now
            s = trainer.summary()
            u = trainer.history["updates"][-1] if trainer.history["updates"] else {}
            print(f"steps {s['steps']:>9,d}  eps {s['episodes']:>6d}  upd {s['updates']:>4d}  "
                  f"success {s['success_rate']:5.1%}  return {s['mean_return']:7.2f}  "
                  f"collide {s['collision_rate']:5.1%}  inside {s['mean_inside']:4.2f}  "
                  f"angle {s['mean_angle_err']:5.1f}  diff {s['difficulty']:.1f}  "
                  f"ent {u.get('entropy', 0):.2f}  sps {s['sps']:,.0f}", flush=True)
        if save_path and now - last_save >= save_every:
            last_save = now
            trainer.save(save_path)
    if save_path:
        trainer.save(save_path)
    return trainer.summary()
