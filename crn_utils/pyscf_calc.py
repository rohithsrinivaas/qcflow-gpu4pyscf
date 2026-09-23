"""GPU4PySCF compute layer: mean-field construction and geometry optimization.
No Hessian/frequency/thermo analysis anywhere in this module.

Replaces atomate.qchem.fireworks.core.{FrequencyFlatteningOptimizeFW,SinglePointFW}.
Adapted from the reference implementation at /data/shared/rohith_data/rishabh
(pyscf2ase.py, run_opt_vacuum.py, resp_charges/*/run_resp_vacuum.py), fixing two
unit bugs there: PySCF energies/gradients/Hessians are in Hartree / Hartree-per-Bohr
[^2], while ASE (and therefore Sella) expects eV / eV-per-Angstrom[^2].

Every speed-related knob below has a Q-Chem counterpart, and the defaults
reproduce this workflow's original behaviour; `config.yaml` documents the
Q-Chem-like alternatives and `benchmark_scf_guess.py` measures them:

    calc_config key      Q-Chem analogue                     default here
    scf_guess            SCF_GUESS=READ between opt cycles   "density" (Q-Chem: MOs -> "mo")
    reuse_mf             (implicit: one process, one mf)     True
    scf_conv_tol[_grad]  SCF_CONVERGENCE (8 for opt)         1e-10 / None
    grid                 XC_GRID (SG-3 for wB97M-V)          unpruned-parent (99,590), NWChem-pruned
    nlc_grid             NL_GRID (SG-1)                      gpu4pyscf default, level 3
    final_grid           (fine grid for the final energy)    None -> same as `grid`
    optimizer            libopt3: BFGS + model Hessian       "sella" (exact Hessian every 20 steps)
    hessian_every        RECOMPUTE_HESSIAN_CYCLES            20
    hessian_grid         (finer grid for second derivatives)  None -> same as `grid`
    guess_energy_guard   (custodian-style sanity check)      0.02 Ha: a seeded SCF that lands
                                                             this far above the previous
                                                             geometry's energy is redone cold
"""

import logging
import os
import tempfile

import numpy as np
import pyscf
from ase import units as ase_units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from gpu4pyscf.dft import gen_grid, rks, uks
from gpu4pyscf.solvent import smd as gpu4pyscf_smd
from pymatgen.io.ase import AseAtomsAdaptor
from pyscf.hessian import thermo as pyscf_thermo
from pyscf.lo import orth as pyscf_orth
from pyscf.scf import addons as scf_addons
from sella import Sella

logger = logging.getLogger(__name__)

HARTREE2EV = ase_units.Hartree
BOHR2ANG = ase_units.Bohr

# Custom SMD solvent (EC:EMC electrolyte blend), derived from the Q-Chem
# custom-solvent descriptor order [Dielec, SolN, SolA, SolB, SolG, SolC, SolH]
# mapped onto gpu4pyscf's [n, n25, alpha, beta, gamma, epsilon, phi, psi]
# (n25 defaults to n, as Q-Chem's custom-solvent input has no SolN25 slot).
gpu4pyscf_smd.solvent_db.setdefault(
    "EC:EMC", [1.415, 1.415, 0.00, 0.735, 20.2, 18.5, 0.00, 0.00]
)

# The grid this workflow has always used: the (99,590) parent of Q-Chem's SG-3,
# with gpu4pyscf's NWChem-style pruning. Q-Chem's SG-3 keeps 30% of that
# parent's points; gpu4pyscf's `{"level": 3}` lands in the same neighbourhood
# (see README), which is what the `grid-l3` benchmark variant measures.
PRODUCTION_GRID = {"atom_grid": (99, 590), "prune": "nwchem"}

_PRUNE_SCHEMES = {
    "nwchem": gen_grid.nwchem_prune,
    "sg1": gen_grid.sg1_prune,
    "treutler": gen_grid.treutler_prune,
    "none": None,
    None: None,
}

SCF_GUESSES = ("atomic", "density", "mo")
OPTIMIZERS = ("sella", "sella-quasi", "geometric")


def _to_numpy(array):
    """CuPy -> NumPy when needed; PySCF's projection/orthogonalization code is CPU-side."""
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def ase_to_pyscf_atom_string(atoms):
    """Format an ASE Atoms object as a PySCF `atom` string."""
    lines = []
    for atom in atoms:
        x, y, z = atom.position
        lines.append(f"{atom.symbol:<2s} {x:>18.10f} {y:>18.10f} {z:>18.10f}")
    return "\n".join(lines)


def _attach_solvent(mf, smd_solvent):
    if not smd_solvent:
        return mf
    mf = mf.SMD()
    mf.with_solvent.method = "SMD"
    mf.with_solvent.solvent = smd_solvent
    mf.with_solvent.lebedev_order = 29
    return mf


def configure_grids(grids, spec):
    """Apply a grid spec to a gpu4pyscf Grids object. `spec` is either
    `{"atom_grid": (n_radial, n_angular)}` (one Lebedev grid for every atom,
    Q-Chem's XC_GRID=XY form) or `{"level": n}` (gpu4pyscf's per-element
    presets, its analogue of the SG-n grids), or `{"level": n, "atom_grid":
    {"H": (n_radial, n_angular)}}` to override the preset for the listed
    elements only, optionally with
    `"prune": "nwchem" | "sg1" | "treutler" | "none"`. None leaves the
    gpu4pyscf defaults in place."""
    if not spec:
        return grids
    atom_grid = spec.get("atom_grid")
    if isinstance(atom_grid, dict):
        # Per-element grids, e.g. {"H": [40, 194]}: listed elements get these
        # (n_radial, n_angular); every other element falls back to `level`
        # (or to a "default" entry). Q-Chem's XC_GRID=0 -- "SG-0 for H, C, N,
        # O; SG-1 for the rest" -- is the same idea.
        grids.atom_grid = {sym: tuple(rad_ang) for sym, rad_ang in atom_grid.items()}
    elif atom_grid:
        grids.atom_grid = tuple(atom_grid)
    else:
        grids.atom_grid = {}
    if "level" in spec:
        grids.level = int(spec["level"])
    if "prune" in spec:
        grids.prune = _PRUNE_SCHEMES[spec["prune"]]
    return grids


def build_mf(mol, xc, smd_solvent=None, dispersion=None, grid=None, nlc_grid=None,
             scf_conv_tol=1e-10, scf_conv_tol_grad=None, max_scf_cycles=200):
    """Build a GPU4PySCF mean-field object for an already-built pyscf.Mole.

    `dispersion` defaults to None: wB97M-V carries VV10 non-local correlation,
    so a D3 correction on top would double count (and dftd3 has no parameters
    for it anyway). `grid` None means PRODUCTION_GRID; `nlc_grid` is the
    (usually coarser) grid for the VV10 term, Q-Chem's NL_GRID.
    """
    mol.verbose = 1
    mf = rks.RKS(mol, xc=xc) if mol.spin == 0 else uks.UKS(mol, xc=xc)
    mf = mf.density_fit()
    mf.disp = dispersion
    configure_grids(mf.grids, PRODUCTION_GRID if grid is None else grid)
    configure_grids(mf.nlcgrids, nlc_grid)
    mf.conv_tol = scf_conv_tol
    if scf_conv_tol_grad is not None:
        mf.conv_tol_grad = scf_conv_tol_grad
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = 1e-14
    mf.small_rho_cutoff = 1e-10
    return _attach_solvent(mf, smd_solvent)


def mf_kwargs_from_config(calc_config, final=False):
    """`build_mf` keyword arguments for a calc_config. With `final=True` the
    `final_grid` (if any) replaces `grid`: that is the grid the reported
    energy is evaluated on, while `grid` drives the optimization."""
    grid = calc_config.get("grid")
    if final and calc_config.get("final_grid"):
        grid = calc_config["final_grid"]
    return {
        "xc": calc_config["xc"],
        "smd_solvent": calc_config.get("smd_solvent"),
        "dispersion": calc_config.get("dispersion"),
        "grid": grid,
        "nlc_grid": calc_config.get("nlc_grid"),
        "scf_conv_tol": calc_config.get("scf_conv_tol", 1e-10),
        "scf_conv_tol_grad": calc_config.get("scf_conv_tol_grad"),
    }


def build_mol(atom_string, basis, charge, spin, max_memory=32000):
    return pyscf.M(atom=atom_string, basis=basis, charge=charge, spin=spin,
                   max_memory=max_memory)


class GPU4PySCFCalculator(Calculator):
    """ASE Calculator that (re)builds a GPU4PySCF mean-field object at each
    geometry the optimizer asks for, converting PySCF's Hartree /
    Hartree-per-Bohr units to ASE's eV / eV-per-Angstrom.

    `scf_guess` decides where each SCF starts from:
      "atomic"   the atomic-density (SAD) guess every time;
      "density"  the previous geometry's converged density matrix projected
                 onto the new AO basis (`project_dm_nr2nr`);
      "mo"       Q-Chem's scheme: the previous geometry's MO coefficients
                 carried over and re-orthogonalized in the new AO basis, the
                 density rebuilt from them -- exactly idempotent, which a
                 projected density matrix is not.
    The converged `mf` for the most recent geometry is kept on `self.mf` so
    that other SCF consumers at the same geometry (Sella's analytic Hessian,
    the final single point) can reuse it.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, mf_builder, charge, spin, basis, scf_guess="density",
                 reuse_dm=None, guess_energy_guard=0.02, **kwargs):
        super().__init__(**kwargs)
        if reuse_dm is not None:   # the pre-`scf_guess` spelling of the same choice
            scf_guess = "density" if reuse_dm else "atomic"
        if scf_guess not in SCF_GUESSES:
            raise ValueError(f"scf_guess must be one of {SCF_GUESSES}, got {scf_guess!r}")
        self.mf_builder = mf_builder
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.scf_guess = scf_guess
        self.guess_energy_guard = guess_energy_guard
        self.mf = None
        self.n_scf_restarts = 0   # SCFs that only converged after falling back
        self.scf_restart_log = []  # which fallback rescued each of them
        self.n_guess_rejections = 0   # seeded SCFs redone cold because they landed too high


    def initial_guess(self, mol):
        """Starting density for an SCF on `mol` (NumPy array for
        `mf.kernel(dm0=...)`), or None for the default guess when reuse is
        off / nothing has converged yet / the projection failed."""
        if self.scf_guess == "atomic" or self.mf is None or not self.mf.converged:
            return None
        try:
            if self.scf_guess == "mo":
                return self._carry_orbitals(mol)
            return self._project_density(mol)
        except Exception:
            logger.warning("Could not build the %s SCF guess at the new geometry; "
                           "using the default guess", self.scf_guess, exc_info=True)
            return None

    def _project_density(self, mol):
        dm_prev = _to_numpy(self.mf.make_rdm1())
        return scf_addons.project_dm_nr2nr(self.mf.mol, dm_prev, mol)

    def _carry_orbitals(self, mol):
        """Occupied orbitals from the previous geometry, re-orthogonalized
        (Lowdin) in the AO basis at the new geometry, then the density from
        them with the previous occupations."""
        mo = _to_numpy(self.mf.mo_coeff)
        occ = _to_numpy(self.mf.mo_occ)
        overlap = mol.intor("int1e_ovlp")

        def carry(coeff):
            moved = scf_addons.project_mo_nr2nr(self.mf.mol, coeff, mol)
            return pyscf_orth.vec_lowdin(moved, overlap)

        if mo.ndim == 3:   # unrestricted: one block per spin
            mo_new = np.array([carry(mo[0]), carry(mo[1])])
        else:
            mo_new = carry(mo)
        return _to_numpy(self.mf.make_rdm1(mo_new, occ))

    def converged_mf(self, atoms):
        """The converged mean-field object if it belongs to exactly this
        geometry (positions, numbers, cell, pbc unchanged), else None."""
        if self.mf is None or self.atoms is None or not self.mf.converged:
            return None
        if self.check_state(atoms):   # non-empty list of system_changes
            return None
        return self.mf

    def run_scf(self, mol, dm0=None, **mf_overrides):
        """Build a mean-field object for `mol` (with any `build_mf` overrides,
        e.g. `grid=`), converge it from `dm0` or the configured guess, and
        make it the current `self.mf`.

        A reused guess occasionally leads DIIS astray after a large geometry
        step (seen on radical anions), so a failed SCF is retried the way
        Q-Chem's custodian handlers do it: first from the atomic-density
        guess, then with a level shift on top. Only then is it an error."""
        mf = self.mf_builder(mol, **mf_overrides)
        if dm0 is None:
            dm0 = self.initial_guess(mol)
        previous = self.mf
        self.mf = mf
        mf.kernel(dm0=dm0)
        if mf.converged and dm0 is not None and self._seeded_solution_suspect(mf, previous):
            # A projected guess can converge to a different SCF solution
            # (seen: +49 kcal/mol on a radical anion, with <S^2> unchanged, so
            # spin contamination cannot catch it). No optimizer step raises the
            # energy that much; a cold start decides which solution is real.
            logger.warning("Seeded SCF landed %.1f kcal/mol above the previous geometry; "
                           "redoing it from the atomic-density guess",
                           (mf.e_tot - previous.e_tot) * 627.5094740631)
            cold = self.mf_builder(mol, **mf_overrides)
            cold.kernel()
            if cold.converged and cold.e_tot < mf.e_tot:
                self.n_guess_rejections += 1
                self.mf = mf = cold
        if mf.converged:
            return mf

        # Escalating fallbacks, cheapest first; each is a fresh mean field.
        # Order and contents come from scf_failure_probe.py on the three
        # strained radical anions whose SCF died in production (job 58782956):
        # Huckel guess converged 2/3 in ~10 s; second-order SCF (Newton)
        # converged 3/3 and found the lowest state each time (~40-80 s);
        # a wider DIIS space 1/3; level shifts, damping and fractional-
        # occupation smearing 0/3 (smearing converged once, to a state
        # 21 kcal/mol too high), so those are gone.
        fallbacks = []
        if dm0 is not None:
            fallbacks.append(("atomic-density guess", self._scf_plain, {}))
        fallbacks += [
            ("Huckel guess", self._scf_plain, {"init_guess": "huckel"}),
            ("second-order SCF (Newton)", self._scf_newton, {}),
            ("DIIS space 12 from cycle 5", self._scf_plain, {"diis_space": 12, "diis_start_cycle": 5}),
        ]
        tried = []
        for label, strategy, settings in fallbacks:
            logger.warning("SCF did not converge; retrying with %s", label)
            tried.append(label)
            try:
                mf = strategy(mol, mf_overrides, settings)
            except Exception:   # noqa: BLE001 -- a failing remedy just moves on to the next
                logger.warning("fallback '%s' raised; moving on", label, exc_info=True)
                continue
            if mf is not None and mf.converged:
                self.mf = mf
                self.n_scf_restarts += 1
                self.scf_restart_log.append(label)
                return mf
        raise RuntimeError("SCF did not converge after every fallback: " + "; ".join(tried))

    def _scf_plain(self, mol, mf_overrides, settings):
        mf = self.mf_builder(mol, **mf_overrides)
        for name, value in settings.items():
            setattr(mf, name, value)
        mf.kernel()
        return mf

    def _scf_newton(self, mol, mf_overrides, settings):
        """Second-order (co-iterative augmented Hessian) SCF, then a short
        plain SCF from its density so the returned object is an ordinary mean
        field for the gradient and Hessian code."""
        mf = self.mf_builder(mol, **mf_overrides)
        newton = mf.newton()
        newton.max_cycle = 50
        newton.kernel()
        if not newton.converged:
            return None
        final = self.mf_builder(mol, **mf_overrides)
        final.kernel(dm0=_to_numpy(newton.make_rdm1()))
        return final

    def _seeded_solution_suspect(self, mf, previous):
        """True when a seeded SCF sits implausibly far above the previous
        geometry's energy. Only meaningful between consecutive geometries of
        one optimization (same molecule, same settings)."""
        if not self.guess_energy_guard or previous is None or not getattr(previous, "converged", False):
            return False
        if getattr(mf, "mol", None) is not None and getattr(previous, "mol", None) is not None \
                and mf.mol.nelectron != previous.mol.nelectron:
            return False
        return float(mf.e_tot) - float(previous.e_tot) > self.guess_energy_guard

    def cold_scf(self, mol):
        """A throwaway mean field converged from the atomic-density guess,
        leaving `self.mf` alone -- what the Hessian and the final single
        point did before `reuse_mf`; kept for the `legacy` benchmark variant."""
        mf = self.mf_builder(mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        return mf

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        mol = build_mol(ase_to_pyscf_atom_string(atoms), self.basis,
                         self.charge, self.spin)
        mf = self.run_scf(mol)

        grad_ha_bohr = mf.nuc_grad_method().kernel()
        self.results["energy"] = float(mf.e_tot) * HARTREE2EV
        self.results["forces"] = -grad_ha_bohr * HARTREE2EV / BOHR2ANG


def hessian_in_ase_units(mf):
    """Analytic Hessian from a converged `mf`, in eV/Angstrom^2 for Sella."""
    h = mf.Hessian()
    h.auxbasis_response = 2
    h_ha_bohr2 = h.kernel()
    natm = h_ha_bohr2.shape[0]
    h_ha_bohr2 = h_ha_bohr2.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    return h_ha_bohr2 * HARTREE2EV / BOHR2ANG ** 2


def check_spin_contamination(mf, spin, significance_threshold=0.1):
    """Compare computed <S^2> to the ideal value S(S+1) for an unrestricted
    calculation. Returns None for closed-shell (spin == 0) calculations.

    `significance_threshold` is a fractional deviation from the ideal <S^2>
    (10% is a common rule-of-thumb cutoff for flagging contaminated radicals).
    """
    if spin == 0:
        return None

    s2, multiplicity_computed = mf.spin_square()
    s = spin / 2.0
    ideal_s2 = s * (s + 1)
    deviation = abs(s2 - ideal_s2)
    significant = deviation > significance_threshold * max(ideal_s2, 1e-8)

    if significant:
        logger.warning(
            "Spin contamination: <S^2>=%.4f vs ideal %.4f (multiplicity_computed=%.3f)",
            s2, ideal_s2, multiplicity_computed,
        )

    return {
        "s2": float(s2),
        "ideal_s2": float(ideal_s2),
        "multiplicity_computed": float(multiplicity_computed),
        "significant": bool(significant),
    }


def _molecule_to_atoms(molecule):
    return AseAtomsAdaptor.get_atoms(molecule)


def _atoms_to_molecule(atoms, charge, spin_multiplicity):
    mol = AseAtomsAdaptor.get_molecule(atoms)
    mol.set_charge_and_spin(charge=charge, spin_multiplicity=spin_multiplicity)
    return mol


def _make_ase_engine(atoms, trajectory):
    """geomeTRIC engine that evaluates energy and gradient through an ASE
    Atoms object, so our GPU4PySCFCalculator, its SCF guess and the
    benchmark's instrumentation are shared with the Sella path. Built here
    so `geometric` stays an optional import."""
    import geometric.engine
    import geometric.molecule

    class ASEEngine(geometric.engine.Engine):
        def __init__(self):
            molecule = geometric.molecule.Molecule()
            molecule.elem = list(atoms.get_chemical_symbols())
            molecule.xyzs = [atoms.get_positions().copy()]   # Angstrom
            super().__init__(molecule)
            self.ncalls = 0

        def calc_new(self, coords, dirname):
            # geomeTRIC works in Bohr / Hartree; ASE in Angstrom / eV.
            atoms.set_positions(coords.reshape(-1, 3) * BOHR2ANG)
            energy = atoms.get_potential_energy() / HARTREE2EV
            gradient = -atoms.get_forces() * BOHR2ANG / HARTREE2EV
            self.ncalls += 1
            if trajectory is not None:
                trajectory.write(atoms)
            return {"energy": energy, "gradient": gradient.ravel()}

    return ASEEngine()


def _optimize_geometric(atoms, fmax, max_steps, trajectory_path, options):
    """Minimize with geomeTRIC: BFGS from a model Hessian in TRIC internal
    coordinates with RFO steps -- the same design as Q-Chem's libopt3, and
    what pyscf's own `geomopt` uses. No analytic Hessian is ever computed.

    Convergence: Sella stops on the maximum force alone, so `fmax` is mapped
    onto geomeTRIC's `convergence_gmax` and `convergence_grms` (grms <= gmax
    always, which leaves the maximum-force test in charge); geomeTRIC's
    displacement and energy criteria stay at their defaults. `options` is
    passed straight to geomeTRIC, e.g. {"qccnv": True} for Q-Chem's own
    convergence semantics or {"coordsys": "dlc"}.
    Returns (converged, number of energy+gradient evaluations)."""
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError
    from pyscf.geomopt import geometric_solver   # ships the log.ini geomeTRIC insists on

    fmax_au = fmax * BOHR2ANG / HARTREE2EV
    kwargs = {
        "convergence_gmax": fmax_au,
        "convergence_grms": fmax_au,
        "maxiter": max_steps,
        "logIni": os.path.join(os.path.dirname(geometric_solver.__file__), "log.ini"),
    }
    kwargs.update(options or {})

    trajectory = Trajectory(trajectory_path, "w") if trajectory_path else None
    engine = _make_ase_engine(atoms, trajectory)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:   # geomeTRIC's log, .tmp dir, xyz
            geometric.optimize.run_optimizer(customengine=engine,
                                             input=os.path.join(tmpdir, "geomopt"),
                                             **kwargs)
        converged = True
    except GeomOptNotConvergedError:
        converged = False
    finally:
        if trajectory is not None:
            trajectory.close()
    return converged, engine.ncalls


def optimize_and_analyze(molecule, charge, spin_multiplicity, calc_config,
                          fmax=0.01, max_steps=500, order=0, trajectory_path=None):
    """Geometry optimization -> final energy. No post-optimization
    Hessian/frequency/thermo analysis (the default Sella setup still uses an
    analytic Hessian internally during optimization -- see `hessian_function`
    below -- that's a distinct thing from post-optimization frequency/thermo
    analysis).

    Mirrors atomate's FrequencyFlatteningOptimizeFW + SinglePointFW pair: order=0
    drives the minimizer to a true minimum (no imaginary modes), replacing
    Q-Chem's flatten-and-restart loop.
    `calc_config`: {"xc", "basis", "smd_solvent", "dispersion"} plus the
    optional speed knobs listed in the module docstring.
    `trajectory_path`: if given, one ASE trajectory frame (with energy and
    forces) per optimization step is written to this path.
    """
    spin = spin_multiplicity - 1  # pyscf convention: 2S, not 2S+1
    basis = calc_config["basis"]

    scf_guess = calc_config.get("scf_guess")
    if scf_guess is None:   # pre-`scf_guess` configs spell it `reuse_dm`
        scf_guess = "density" if calc_config.get("reuse_dm", True) else "atomic"
    reuse_mf = calc_config.get("reuse_mf", True)
    optimizer = calc_config.get("optimizer", "sella")
    hessian_every = calc_config.get("hessian_every", 20)
    final_grid = calc_config.get("final_grid")
    hessian_grid = calc_config.get("hessian_grid")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")

    mf_kwargs = mf_kwargs_from_config(calc_config)

    def mf_builder(mol, **overrides):
        return build_mf(mol, **{**mf_kwargs, **overrides})

    atoms = _molecule_to_atoms(molecule)
    calc = GPU4PySCFCalculator(mf_builder, charge=charge, spin=spin, basis=basis,
                               scf_guess=scf_guess,
                               guess_energy_guard=calc_config.get("guess_energy_guard", 0.02))
    atoms.calc = calc

    def mol_at(atoms_):
        return build_mol(ase_to_pyscf_atom_string(atoms_), basis, charge, spin)

    def converged_mf_at(atoms_):
        """Converged mf at `atoms_` geometry. With `reuse_mf`: the
        calculator's own if it was just evaluated there (the optimizer always
        evaluates energy/forces before it asks for a Hessian), otherwise a
        fresh SCF from the configured guess. Without: a cold throwaway SCF,
        as before `reuse_mf` existed."""
        if not reuse_mf:
            return calc.cold_scf(mol_at(atoms_))
        mf = calc.converged_mf(atoms_)
        if mf is None:
            mf = calc.run_scf(mol_at(atoms_))
        return mf

    def hessian_function(atoms_):
        mf = converged_mf_at(atoms_)
        if hessian_grid:
            # Second derivatives are far more grid-sensitive than forces (a
            # sparse grid can turn a soft mode imaginary), so the Hessian may
            # be evaluated on a finer grid than the one driving the forces:
            # a few SCF cycles from the converged density, then `self.mf` is
            # put back so the next step's guess stays on the force grid.
            force_mf = calc.mf
            mf = calc.run_scf(mol_at(atoms_), dm0=_to_numpy(mf.make_rdm1()), grid=hessian_grid)
            calc.mf = force_mf
        return hessian_in_ase_units(mf)

    if trajectory_path is not None:
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)

    if optimizer == "geometric":
        converged, n_steps = _optimize_geometric(atoms, fmax, max_steps, trajectory_path,
                                                 calc_config.get("geomopt"))
    else:
        # "sella": Sella's model internal Hessian to start, refreshed with the
        # analytic Hessian every `hessian_every` steps. "sella-quasi": the same
        # optimizer with no analytic Hessian at all -- model Hessian plus
        # quasi-Newton updates, as Q-Chem's libopt3 does for minimizations.
        exact_hessian = optimizer == "sella"
        opt = Sella(atoms, order=order, gamma=0.1, delta0=0.1, threepoint=True,
                    diag_every_n=hessian_every if exact_hessian else None,
                    nsteps_per_diag=10, eta=1e-6, internal=True,
                    hessian_function=hessian_function if exact_hessian else None,
                    trajectory=trajectory_path)
        for _ in opt.irun(fmax=fmax, steps=max_steps):
            pass
        converged = opt.converged()
        n_steps = opt.nsteps

    optimized_molecule = _atoms_to_molecule(atoms, charge, spin_multiplicity)
    mf = converged_mf_at(atoms)
    final_state_check = None
    if scf_guess != "atomic" and calc.guess_energy_guard:
        # The step-to-step guard cannot see a wrong SCF state entered during
        # the steep early descent (the energy still falls step to step), and
        # that state is then inherited to the end -- seen: +99 kcal/mol carried
        # to convergence on a radical cation. One cold SCF at the final
        # geometry settles it; the reported energy is the lower solution.
        cold = calc.cold_scf(mol_at(atoms))
        gap = float(mf.e_tot) - float(cold.e_tot)
        final_state_check = {"seeded_minus_cold_Ha": gap,
                             "wrong_state": gap > calc.guess_energy_guard}
        if final_state_check["wrong_state"]:
            logger.warning("The seeded SCF trajectory ended %.1f kcal/mol above a cold SCF at the "
                           "final geometry: it was on a wrong electronic state. Reporting the "
                           "cold solution; the geometry itself was optimized on the wrong state.",
                           gap * 627.5094740631)
            mf = cold
    if final_grid:
        # The optimization ran on `grid`; the reported energy comes from the
        # finer `final_grid`, seeded with the converged density so it is a
        # few cycles rather than a fresh SCF.
        mf = calc.run_scf(mol_at(atoms), dm0=_to_numpy(mf.make_rdm1()), grid=final_grid)
    energy_ha = float(mf.e_tot)

    spin_contamination = check_spin_contamination(mf, spin)

    result = {
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "initial_molecule": molecule.as_dict(),
        "molecule": optimized_molecule.as_dict(),
        "energy_Ha": energy_ha,
        "converged": bool(converged),
        "n_opt_steps": int(n_steps),
        "n_scf_restarts": calc.n_scf_restarts,
        "scf_restart_log": list(calc.scf_restart_log),
        "final_state_check": final_state_check,
        "n_guess_rejections": calc.n_guess_rejections,
        "basis": basis,
        "xc": calc_config["xc"],
        "smd_solvent": calc_config.get("smd_solvent"),
        "spin_contamination": spin_contamination,
        "settings": {
            "scf_guess": scf_guess,
            "reuse_mf": bool(reuse_mf),
            "optimizer": optimizer,
            "hessian_every": hessian_every if optimizer == "sella" else None,
            "grid": mf_kwargs["grid"] or PRODUCTION_GRID,
            "final_grid": final_grid,
            "hessian_grid": hessian_grid if optimizer == "sella" else None,
            "nlc_grid": mf_kwargs["nlc_grid"],
            "scf_conv_tol": mf_kwargs["scf_conv_tol"],
            "scf_conv_tol_grad": mf_kwargs["scf_conv_tol_grad"],
        },
        "point_group": None,
        "rotational_symmetry_number": None,
    }

    return result


def run_molecule_job(molecule, charge, calc_config, trajectory_path=None):
    """Dispatch to single_point (monoatomic) or optimize_and_analyze (otherwise),
    deriving spin multiplicity from electron count the same way `1-optimize-SP.py`
    always has. `molecule` may already carry a nonzero `.charge` (fragment/
    recombination candidates do) or be neutral (fresh xyz input) -- recover the
    neutral electron count first so `charge` is only ever applied once.
    `trajectory_path`: ASE trajectory output path for optimize_and_analyze; unused
    for single_point (monoatomic species -- no optimization to trace).
    """
    nelectrons_neutral = molecule.nelectrons + molecule.charge
    nelectrons = nelectrons_neutral - charge
    spin_multiplicity = 1 if nelectrons % 2 == 0 else 2
    molecule = molecule.copy()
    molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin_multiplicity)

    if molecule.num_sites == 1:
        return single_point(molecule, charge, spin_multiplicity, calc_config)
    return optimize_and_analyze(molecule, charge, spin_multiplicity, calc_config,
                                 trajectory_path=trajectory_path)


def single_point(molecule, charge, spin_multiplicity, calc_config):
    """SCF only, no optimization/Hessian (monoatomic species). Evaluated on
    `final_grid` when one is configured -- a single point is a final energy."""
    spin = spin_multiplicity - 1
    basis = calc_config["basis"]

    mol = build_mol(ase_to_pyscf_atom_string(_molecule_to_atoms(molecule)),
                     basis, charge, spin)
    mf = build_mf(mol, **mf_kwargs_from_config(calc_config, final=True))
    energy_ha = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    spin_contamination = check_spin_contamination(mf, spin)
    sigma_r = pyscf_thermo.rotational_symmetry_number(mol)

    result = {
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "initial_molecule": molecule.as_dict(),
        "molecule": molecule.as_dict(),
        "energy_Ha": energy_ha,
        "converged": True,
        "basis": basis,
        "xc": calc_config["xc"],
        "smd_solvent": calc_config.get("smd_solvent"),
        "spin_contamination": spin_contamination,
        "point_group": "Kh",
        "rotational_symmetry_number": sigma_r,
    }

    return result
