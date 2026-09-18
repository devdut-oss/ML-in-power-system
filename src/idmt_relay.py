"""
idmt_relay.py
==============
IEC 60255-151 IDMT (Inverse Definite Minimum Time) overcurrent relay model
and time-graded coordination for the 132 kV radial system.

IEC standard inverse-time characteristic:

    t_op = TMS * k / ( (I/Is)^alpha - 1 )        for I > Is

    curve            k       alpha
    -------------  ------   ------
    Standard (SI)   0.14     0.02      <- used here (Indian utility default)
    Very inverse    13.5     1.0
    Extremely inv.  80.0     2.0

where  I  = measured phase current (CT secondary referred, or primary amps
            with Is in primary amps - consistent units),
       Is = pickup (plug) setting,
       TMS = time multiplier setting (0.025 ... 1.2).

COORDINATION PHILOSOPHY (time grading, radial feeder):
    * R2 (downstream, Bus B) protects Line 2 -> fastest possible: TMS = 0.05.
    * R1 (upstream, Bus A) protects Line 1 AND backs up R2. For a fault just
      beyond Bus B, R1 must be slower than R2 by the Coordination Time
      Interval  CTI = 0.3 s  (breaker time + relay overshoot + margin).
    * Pickup: Is = 1.25 x max load current, below min fault current.

The `coordinate()` routine below computes R1's TMS automatically from the
worst-case grading point (maximum fault current at Bus B busbar), exactly
the manual procedure a protection engineer performs.
"""

import numpy as np

CURVES = {
    'SI': (0.14, 0.02),      # standard inverse  3/10
    'VI': (13.5, 1.00),      # very inverse
    'EI': (80.0, 2.00),      # extremely inverse
}


class IDMTRelay:
    def __init__(self, name, pickup_A, TMS, curve='SI', CT_ratio=400.0):
        self.name = name
        self.Is = pickup_A            # pickup in PRIMARY amps
        self.TMS = TMS
        self.k, self.alpha = CURVES[curve]
        self.curve = curve
        self.CT = CT_ratio

    def trip_time(self, I_primary):
        """Operating time in seconds; np.inf if below pickup.

        Vectorised: accepts scalar or ndarray of primary current magnitudes.
        """
        I = np.asarray(I_primary, dtype=float)
        M = I / self.Is                               # plug setting multiplier
        with np.errstate(divide='ignore', invalid='ignore'):
            t = self.TMS * self.k / (np.power(M, self.alpha) - 1.0)
        t = np.where(M > 1.0, t, np.inf)
        # numerical relays clamp minimum operating time (~2 cycles)
        t = np.maximum(t, 0.04)
        return t if t.shape else float(t)

    def picks_up(self, I_primary):
        return np.asarray(I_primary) > self.Is


def coordinate(ps, CTI=0.3, verbose=True):
    """Set both relays for the given PowerSystem `ps`.

    Returns (R1, R2) coordinated IDMTRelay objects.
    """
    I_load = abs(ps.load_current(1.0))

    # ---- pickups: 125 % of max load through each relay ----
    Is2 = 1.25 * I_load                    # R2 carries the full load too (radial)
    Is1 = 1.25 * I_load * 1.15             # R1 slightly higher for selectivity

    R2 = IDMTRelay('R2', Is2, 0.05, 'SI', ps.CT_R2)

    # ---- grading point: max 3-phase fault at Bus B busbar (just after R2) ----
    fb = ps.simulate_fault('ABC', 2, 0.001, 0.0)
    If_gp = fb['R2_Ia']
    t_R2 = R2.trip_time(If_gp)
    t_R1_req = t_R2 + CTI

    # solve required TMS for R1 at grading point
    k, alpha = CURVES['SI']
    M1 = If_gp / Is1
    TMS1 = t_R1_req * (M1**alpha - 1.0) / k
    TMS1 = float(np.clip(np.round(TMS1, 3), 0.025, 1.2))
    R1 = IDMTRelay('R1', Is1, TMS1, 'SI', ps.CT_R1)

    if verbose:
        print(f"Load current            : {I_load:7.1f} A")
        print(f"R2: Is = {Is2:6.1f} A, TMS = {R2.TMS}")
        print(f"Grading fault @ Bus B   : {If_gp:7.1f} A -> "
              f"t_R2 = {t_R2:.3f} s, R1 must be >= {t_R1_req:.3f} s")
        print(f"R1: Is = {Is1:6.1f} A, TMS = {TMS1}")
        # verify at a remote zone-2 fault as well
        fr = ps.simulate_fault('ABC', 2, ps.L2 * 0.99, 0.0)
        print(f"Check remote end L2     : t_R2 = {R2.trip_time(fr['R2_Ia']):.3f} s, "
              f"t_R1 = {R1.trip_time(fr['R1_Ia']):.3f} s "
              f"(margin {R1.trip_time(fr['R1_Ia']) - R2.trip_time(fr['R2_Ia']):.3f} s)")
    return R1, R2


def relay_decision(R1, R2, meas, zone, ftype):
    """Ground-truth protection decision for one fault case.

    Returns dict:
      correct_relay : 'R1' | 'R2' | 'NONE'  (which relay SHOULD clear it)
      t_R1, t_R2    : IDMT operating times on the max faulted-phase current
      coordinated   : True if the time grading held (t_backup - t_primary >= 0.2)
    """
    Imax_R1 = max(meas['R1_Ia'], meas['R1_Ib'], meas['R1_Ic'])
    Imax_R2 = max(meas['R2_Ia'], meas['R2_Ib'], meas['R2_Ic'])
    t1 = R1.trip_time(Imax_R1)
    t2 = R2.trip_time(Imax_R2)

    if ftype == 'NONE':
        correct = 'NONE'
        coordinated = np.isinf(t1) and np.isinf(t2)
    elif zone == 1:
        correct = 'R1'                       # R2 must NOT see zone-1 faults
        coordinated = np.isfinite(t1)
    else:
        correct = 'R2'                       # R1 is remote backup only
        coordinated = (np.isfinite(t2) and
                       (np.isinf(t1) or (t1 - t2) >= 0.2))
    return {'correct_relay': correct,
            't_R1': t1 if np.isfinite(t1) else -1.0,
            't_R2': t2 if np.isfinite(t2) else -1.0,
            'coordinated': bool(coordinated)}


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from power_system import PowerSystem
    ps = PowerSystem()
    R1, R2 = coordinate(ps)
    # spot checks
    for ftype, zone, d, rf in [('ABC', 1, 25, 0), ('AG', 2, 20, 5), ('NONE', 1, 0, 0)]:
        m = ps.simulate_fault(ftype, zone, max(d, 0.001), rf)
        dec = relay_decision(R1, R2, m, zone, ftype)
        print(f"{ftype:>4} z{zone} d={d:>2}km Rf={rf:>2} -> correct={dec['correct_relay']:>4} "
              f"t_R1={dec['t_R1']:6.3f} t_R2={dec['t_R2']:6.3f} coord={dec['coordinated']}")
