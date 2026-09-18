"""
power_system.py
================
Transmission-line model and short-circuit (fault) analysis engine using the
method of SYMMETRICAL COMPONENTS (Fortescue, 1918) - the same phasor-domain
mathematics used inside commercial tools (ETAP, DIgSILENT, MATLAB/Simulink
phasor mode).

SYSTEM UNDER STUDY (single-line diagram)
----------------------------------------
    Grid                Line 1 (50 km)            Line 2 (40 km)
   Source ----[Bus A]====================[Bus B]==================[Bus C]
   3500 MVA     |R1 (IDMT OC relay,        |R2 (IDMT OC relay,      |
                |   CT 400/1)              |   CT 300/1)            Load
                                                              40 MVA, 0.85 pf

    Zone 1 : Line 1  -> primary protection R1
    Zone 2 : Line 2  -> primary protection R2, remote backup R1

VOLTAGE LEVEL : 132 kV line-to-line, 50 Hz  (Indian sub-transmission level,
where IDMT overcurrent relays are the standard primary/backup protection).

FAULT TYPES MODELLED (10 shunt faults + healthy state):
    L-G   : AG, BG, CG          (single line to ground, ~80 % of real faults)
    L-L   : AB, BC, CA          (line to line)
    L-L-G : ABG, BCG, CAG       (double line to ground)
    L-L-L : ABC                 (balanced three phase, most severe)

The sequence-network interconnections implemented below are the classical
textbook results (see e.g. Stevenson & Grainger, "Power System Analysis";
Paithankar & Bhide, "Fundamentals of Power System Protection"):

    3-phase (ABC) : I1 = E / (Z1 + Zf)                     ,  I2 = I0 = 0
    L-G  (AG)     : I1 = I2 = I0 = E / (Z1 + Z2 + Z0 + 3Zf)
    L-L  (BC)     : I1 = -I2 = E / (Z1 + Z2 + Zf)          ,  I0 = 0
    L-L-G(BCG)    : I1 = E / ( Z1 + Z2*(Z0+3Zf)/(Z2+Z0+3Zf) )
                    I2 = -I1 * (Z0 + 3Zf) / (Z2 + Z0 + 3Zf)
                    I0 = -I1 *  Z2        / (Z2 + Z0 + 3Zf)

Phase quantities are recovered with  [Iabc] = [A] [I012],
    A = [[1, 1, 1], [1, a^2, a], [1, a, a^2]],  a = 1<120 deg.

Faults on phases other than the reference phase are handled by cyclically
rotating the phase labels (e.g. a BG fault is an AG fault with the phase
sequence rotated by one position).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Fortescue operator and transformation matrix
# ----------------------------------------------------------------------------
a = np.exp(1j * 2 * np.pi / 3)                      # 1 /_ 120deg
A = np.array([[1, 1, 1],
              [1, a**2, a],
              [1, a, a**2]], dtype=complex)          # seq -> phase (012 -> abc)


class PowerSystem:
    """132 kV radial two-section transmission system with fault analysis."""

    def __init__(self):
        # ------------------- base quantities -------------------
        self.f = 50.0                                # Hz
        self.VLL = 132e3                             # line-line volts
        self.Vph = self.VLL / np.sqrt(3)             # phase volts (76.21 kV)

        # ------------------- grid source (Thevenin) -------------------
        # Short-circuit capacity 3500 MVA, X/R = 10
        SCC = 3500e6
        Zs_mag = self.VLL**2 / SCC                   # |Zs1| = 4.98 ohm
        theta = np.arctan(10.0)                      # X/R = 10
        self.Zs1 = Zs_mag * (np.cos(theta) + 1j * np.sin(theta))
        self.Zs2 = self.Zs1                          # neg-seq = pos-seq (static)
        self.Zs0 = 1.5 * self.Zs1                    # typical grounded source

        # ------------------- line parameters (ACSR 'Panther' class) ---------
        # per-km sequence impedances, typical 132 kV single-circuit values
        self.z1 = 0.120 + 0.400j                     # ohm/km  positive seq
        self.z0 = 0.350 + 1.250j                     # ohm/km  zero seq
        self.L1 = 50.0                               # km, Bus A -> Bus B
        self.L2 = 40.0                               # km, Bus B -> Bus C

        # ------------------- load at Bus C -------------------
        self.S_load = 40e6                           # VA
        self.pf = 0.85                               # lagging
        # constant-impedance load model (for pre-fault current)
        phi = np.arccos(self.pf)
        Zld_mag = self.VLL**2 / self.S_load
        self.Z_load = Zld_mag * (np.cos(phi) + 1j * np.sin(phi))

        # CT ratios (primary A / secondary A)
        self.CT_R1 = 400.0
        self.CT_R2 = 300.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def load_current(self, load_factor=1.0):
        """Pre-fault load current phasor (phase A, at nominal voltage).

        Series impedance to the load: source + both lines. Load scaled by
        `load_factor` (0..1.2) via its admittance.
        """
        Zseries = self.Zs1 + self.z1 * (self.L1 + self.L2)
        Zld = self.Z_load / max(load_factor, 1e-6)
        return self.Vph / (Zseries + Zld)

    def _thevenin_to_fault(self, zone, dist_km):
        """Sequence Thevenin impedances from the source up to the fault point.

        zone 1: fault on Line 1, dist_km measured from Bus A.
        zone 2: fault on Line 2, dist_km measured from Bus B.
        (Radial single-source system -> load-side infeed neglected in the
        pure-fault network; load handled by superposition.)
        """
        if zone == 1:
            d = dist_km
            Z1 = self.Zs1 + self.z1 * d
            Z0 = self.Zs0 + self.z0 * d
        else:
            Z1 = self.Zs1 + self.z1 * self.L1 + self.z1 * dist_km
            Z0 = self.Zs0 + self.z0 * self.L1 + self.z0 * dist_km
        Z2 = Z1                                       # lines: Z2 = Z1
        return Z1, Z2, Z0

    @staticmethod
    def _sequence_fault_currents(ftype_ref, E, Z1, Z2, Z0, Rf):
        """Sequence currents AT THE FAULT for reference-phase fault types.

        ftype_ref in {'3PH','LG','LL','LLG'} with phase A (or BC) reference.
        Returns (I0, I1, I2).
        """
        Zf = Rf + 0j
        if ftype_ref == '3PH':
            I1 = E / (Z1 + Zf)
            return 0j, I1, 0j
        if ftype_ref == 'LG':
            I1 = E / (Z1 + Z2 + Z0 + 3 * Zf)
            return I1, I1, I1
        if ftype_ref == 'LL':
            I1 = E / (Z1 + Z2 + Zf)
            return 0j, I1, -I1
        if ftype_ref == 'LLG':
            Zpar = Z2 * (Z0 + 3 * Zf) / (Z2 + Z0 + 3 * Zf)
            I1 = E / (Z1 + Zpar)
            I2 = -I1 * (Z0 + 3 * Zf) / (Z2 + Z0 + 3 * Zf)
            I0 = -I1 * Z2 / (Z2 + Z0 + 3 * Zf)
            return I0, I1, I2
        raise ValueError(ftype_ref)

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    #   fault type string -> (reference network, phase rotation)
    #   rotation r means: simulated phases are rolled so that the reference
    #   phase pattern lands on the requested phases.
    FAULT_MAP = {
        'AG':  ('LG', 0), 'BG':  ('LG', 1), 'CG':  ('LG', 2),
        'BC':  ('LL', 0), 'CA':  ('LL', 1), 'AB':  ('LL', 2),
        'BCG': ('LLG', 0), 'CAG': ('LLG', 1), 'ABG': ('LLG', 2),
        'ABC': ('3PH', 0),
    }

    def simulate_fault(self, ftype, zone, dist_km, Rf, load_factor=1.0):
        """Run one steady-state fault case.

        Parameters
        ----------
        ftype       : one of FAULT_MAP keys, or 'NONE' for healthy operation
        zone        : 1 (fault on Line 1) or 2 (fault on Line 2)
        dist_km     : distance of fault from the zone's sending bus
        Rf          : fault resistance in ohm
        load_factor : per-unit loading of the 40 MVA load

        Returns
        -------
        dict with phasors seen by relays R1 (Bus A) and R2 (Bus B):
        phase current magnitudes/angles, bus voltage magnitudes,
        and sequence current magnitudes - i.e. exactly the quantities a
        numerical relay extracts with a phasor (DFT) estimator.
        """
        E = self.Vph + 0j                            # 1.0 pu prefault, ref phase A
        I_load = self.load_current(load_factor)
        Iabc_load = np.array([I_load,
                              I_load * a**2,
                              I_load * a])           # balanced abc load set

        if ftype == 'NONE':
            Iabc_R1 = Iabc_load
            Iabc_R2 = Iabc_load
            # voltage drop to each bus under load only
            V_A = E - I_load * self.Zs1
            V_B = V_A - I_load * self.z1 * self.L1
            Vabc_A = np.array([V_A, V_A * a**2, V_A * a])
            Vabc_B = np.array([V_B, V_B * a**2, V_B * a])
        else:
            ref, rot = self.FAULT_MAP[ftype]
            Z1, Z2, Z0 = self._thevenin_to_fault(zone, dist_km)
            I0, I1, I2 = self._sequence_fault_currents(ref, E, Z1, Z2, Z0, Rf)

            # In a radial single-source system the entire fault current flows
            # from the source through R1; it also flows through R2 only when
            # the fault is in zone 2.
            Iseq_fault = np.array([I0, I1, I2])
            Iabc_fault = A @ Iseq_fault
            Iabc_fault = np.roll(Iabc_fault, rot)    # move ref phases -> actual

            # sequence voltages at Bus A (behind R1)
            V1_A = E - I1 * self.Zs1
            V2_A = -I2 * self.Zs2
            V0_A = -I0 * self.Zs0
            Vabc_A = np.roll(A @ np.array([V0_A, V1_A, V2_A]), rot)

            if zone == 1:
                # fault current in line 1 only; R2 sees (collapsed) load only
                Iabc_R1 = Iabc_fault + Iabc_load          # superposition
                Iabc_R2 = Iabc_load * 0.3                 # load partially fed
                # Bus B voltage: healthy-line drop beyond the fault is small;
                # approximate Bus B voltage ~ voltage at fault point
                V1_B = E - I1 * (self.Zs1 + self.z1 * dist_km)
                V2_B = -I2 * (self.Zs2 + self.z1 * dist_km)
                V0_B = -I0 * (self.Zs0 + self.z0 * dist_km)
                Vabc_B = np.roll(A @ np.array([V0_B, V1_B, V2_B]), rot)
            else:
                # fault beyond Bus B: full fault current through both relays
                Iabc_R1 = Iabc_fault + Iabc_load
                Iabc_R2 = Iabc_fault + Iabc_load
                V1_B = E - I1 * (self.Zs1 + self.z1 * self.L1)
                V2_B = -I2 * (self.Zs2 + self.z1 * self.L1)
                V0_B = -I0 * (self.Zs0 + self.z0 * self.L1)
                Vabc_B = np.roll(A @ np.array([V0_B, V1_B, V2_B]), rot)

        # ---- assemble measurement dictionary (what the relay's DFT sees) ----
        out = {}
        for name, Iabc in (('R1', Iabc_R1), ('R2', Iabc_R2)):
            out[f'{name}_Ia'] = abs(Iabc[0]); out[f'{name}_Ib'] = abs(Iabc[1])
            out[f'{name}_Ic'] = abs(Iabc[2])
            out[f'{name}_Ia_ang'] = np.angle(Iabc[0], deg=True)
            out[f'{name}_Ib_ang'] = np.angle(Iabc[1], deg=True)
            out[f'{name}_Ic_ang'] = np.angle(Iabc[2], deg=True)
            # sequence components recovered from the phase set
            Iseq = np.linalg.solve(A, Iabc)          # [I0, I1, I2]
            out[f'{name}_I0'] = abs(Iseq[0])
            out[f'{name}_I1'] = abs(Iseq[1])
            out[f'{name}_I2'] = abs(Iseq[2])
        for name, Vabc in (('VA', Vabc_A), ('VB', Vabc_B)):
            out[f'{name}_a'] = abs(Vabc[0]); out[f'{name}_b'] = abs(Vabc[1])
            out[f'{name}_c'] = abs(Vabc[2])
            Vseq = np.linalg.solve(A, Vabc)
            out[f'{name}_V0'] = abs(Vseq[0])
            out[f'{name}_V2'] = abs(Vseq[2])
        return out


if __name__ == '__main__':
    ps = PowerSystem()
    print(f"System: {ps.VLL/1e3:.0f} kV, source Zs1 = {ps.Zs1:.3f} ohm")
    print(f"Full-load current  = {abs(ps.load_current()):8.1f} A")
    # sanity check: bolted 3-phase fault at Bus A busbar
    r = ps.simulate_fault('ABC', 1, 0.001, 0.0)
    print(f"3ph fault @ Bus A  = {r['R1_Ia']:8.1f} A  "
          f"(expect ~ VLL^2/SCC -> I = {3500e6/np.sqrt(3)/132e3:.0f} A)")
    r = ps.simulate_fault('AG', 2, 20.0, 10.0)
    print(f"AG fault mid line2 = R1 Ia {r['R1_Ia']:8.1f} A | "
          f"R2 Ia {r['R2_Ia']:8.1f} A | I0 at R2 {r['R2_I0']:.1f} A")
