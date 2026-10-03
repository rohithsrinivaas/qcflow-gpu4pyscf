"""GPU4PySCF compute layer: mean-field construction and Sella geometry optimization.
No Hessian/frequency/thermo analysis anywhere in this module.

Replaces atomate.qchem.fireworks.core.{FrequencyFlatteningOptimizeFW,SinglePointFW}.
Adapted from the reference implementation at /data/shared/rohith_data/rishabh
(pyscf2ase.py, run_opt_vacuum.py, resp_charges/*/run_resp_vacuum.py), fixing two
unit bugs there: PySCF energies/gradients/Hessians are in Hartree / Hartree-per-Bohr
[^2], while ASE (and therefore Sella) expects eV / eV-per-Angstrom[^2].
"""

import logging
import os
import time

import pyscf
from ase import units as ase_units
from ase.calculators.calculator import Calculator, all_changes
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.solvent import smd as gpu4pyscf_smd
from pymatgen.io.ase import AseAtomsAdaptor
from pyscf.hessian import thermo as pyscf_thermo
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


# CDIIS cycles before run_scf() hands over to SOSCF. Easy SCFs converge in
# 15-50 cycles; ones that don't by then mostly oscillate rather than converge
# slowly, so switching early saves the ~150 wasted cycles of a 200 cap.
CDIIS_MAX_CYCLE = 50
SOSCF_MAX_CYCLE = 50


def build_mf(mol, xc, smd_solvent=None, dispersion="d3bj", grids_level=3,
             scf_conv_tol=1e-10, max_scf_cycles=CDIIS_MAX_CYCLE):
    """Build a GPU4PySCF mean-field object for an already-built pyscf.Mole."""
    mol.verbose = 1
    mf = rks.RKS(mol, xc=xc) if mol.spin == 0 else uks.UKS(mol, xc=xc)
    mf = mf.density_fit()
    mf.disp = dispersion
    mf.grids.level = grids_level
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = scf_conv_tol
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = 1e-14
    mf.small_rho_cutoff = 1e-10
    return _attach_solvent(mf, smd_solvent)


def run_scf(mf, stats=None):
    """Run the SCF with the default CDIIS solver; if that doesn't converge,
    continue from its last orbitals with the second-order (CIAH Newton) SOSCF
    solver, which handles the small-gap charged open-shell cases where DIIS
    oscillates. Returns a converged plain mean-field object (SOSCF mixin
    removed, so gradients/Hessians work as usual) or raises RuntimeError.
    `stats`, if given, counts SCFs and SOSCF fallbacks: {"n_scf", "n_soscf"}.
    """
    if stats is not None:
        stats["n_scf"] = stats.get("n_scf", 0) + 1
    mf.kernel()
    if mf.converged:
        return mf

    logger.warning("CDIIS did not converge in %s cycles; retrying with SOSCF", mf.cycles)
    if stats is not None:
        stats["n_soscf"] = stats.get("n_soscf", 0) + 1
    mf_soscf = mf.newton()
    mf_soscf.max_cycle = SOSCF_MAX_CYCLE
    mf_soscf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf_soscf.converged:
        raise RuntimeError("SCF did not converge (CDIIS, then SOSCF)")
    return mf_soscf.undo_soscf()


def build_mol(atom_string, basis, charge, spin, max_memory=32000):
    return pyscf.M(atom=atom_string, basis=basis, charge=charge, spin=spin,
                   max_memory=max_memory)


class GPU4PySCFCalculator(Calculator):
    """ASE Calculator that (re)builds a GPU4PySCF mean-field object at each
    geometry Sella asks for, converting PySCF's Hartree / Hartree-per-Bohr
    units to ASE's eV / eV-per-Angstrom.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, mf_builder, charge, spin, basis, scf_stats=None, **kwargs):
        super().__init__(**kwargs)
        self.mf_builder = mf_builder
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.scf_stats = scf_stats

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        mol = build_mol(ase_to_pyscf_atom_string(atoms), self.basis,
                         self.charge, self.spin)
        mf = run_scf(self.mf_builder(mol), stats=self.scf_stats)
        energy_ha = mf.e_tot

        grad_ha_bohr = mf.nuc_grad_method().kernel()
        self.results["energy"] = energy_ha * HARTREE2EV
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


def optimize_and_analyze(molecule, charge, spin_multiplicity, calc_config,
                          fmax=0.01, max_steps=500, order=0, trajectory_path=None):
    """Sella geometry optimization -> final energy. No post-optimization
    Hessian/frequency/thermo analysis (Sella still uses an analytic Hessian
    internally during optimization -- see `hessian_function` below -- that's
    a distinct thing from post-optimization frequency/thermo analysis).

    Mirrors atomate's FrequencyFlatteningOptimizeFW + SinglePointFW pair: order=0
    drives Sella's eigenvector-following minimizer to a true minimum (no
    imaginary modes), replacing Q-Chem's flatten-and-restart loop.
    `calc_config`: {"xc": ..., "basis": ..., "smd_solvent": ..., "dispersion": ...}
    `trajectory_path`: if given, Sella writes one ASE trajectory frame (with
    energy and forces) per accepted optimization step to this path.
    """
    spin = spin_multiplicity - 1  # pyscf convention: 2S, not 2S+1
    basis = calc_config["basis"]

    def mf_builder(mol):
        return build_mf(mol, xc=calc_config["xc"],
                         smd_solvent=calc_config.get("smd_solvent"),
                         dispersion=calc_config.get("dispersion"))

    scf_stats = {"n_scf": 0, "n_soscf": 0}

    def hessian_function(atoms_):
        mol = build_mol(ase_to_pyscf_atom_string(atoms_), basis, charge, spin)
        mf = run_scf(mf_builder(mol), stats=scf_stats)
        return hessian_in_ase_units(mf)

    atoms = _molecule_to_atoms(molecule)
    atoms.calc = GPU4PySCFCalculator(mf_builder, charge=charge, spin=spin, basis=basis,
                                     scf_stats=scf_stats)

    if trajectory_path is not None:
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)

    opt = Sella(atoms, order=order, gamma=0.1, delta0=0.1, threepoint=True,
                diag_every_n=20, nsteps_per_diag=10, eta=1e-6, internal=True,
                hessian_function=hessian_function, trajectory=trajectory_path,
                logfile=None)  # per-step logging is too verbose at full-run scale;
                                # callers get a step count/wall time in the result instead
    opt_start = time.time()
    n_steps = 0
    for _ in opt.irun(fmax=fmax, steps=max_steps):
        n_steps += 1
    opt_wall_time_s = time.time() - opt_start
    converged = opt.converged()

    optimized_molecule = _atoms_to_molecule(atoms, charge, spin_multiplicity)
    final_mol = build_mol(ase_to_pyscf_atom_string(atoms), basis, charge, spin)
    mf = run_scf(mf_builder(final_mol), stats=scf_stats)
    energy_ha = mf.e_tot

    spin_contamination = check_spin_contamination(mf, spin)

    result = {
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "initial_molecule": molecule.as_dict(),
        "molecule": optimized_molecule.as_dict(),
        "energy_Ha": energy_ha,
        "converged": bool(converged),
        "basis": basis,
        "xc": calc_config["xc"],
        "smd_solvent": calc_config.get("smd_solvent"),
        "spin_contamination": spin_contamination,
        "point_group": None,
        "rotational_symmetry_number": None,
        "n_opt_steps": n_steps,
        "opt_wall_time_s": opt_wall_time_s,
        "n_scf": scf_stats["n_scf"],
        "n_soscf_fallbacks": scf_stats["n_soscf"],
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
    """SCF only, no optimization/Hessian (monoatomic species)."""
    spin = spin_multiplicity - 1
    basis = calc_config["basis"]

    mol = build_mol(ase_to_pyscf_atom_string(_molecule_to_atoms(molecule)),
                     basis, charge, spin)
    mf = build_mf(mol, xc=calc_config["xc"],
                  smd_solvent=calc_config.get("smd_solvent"),
                  dispersion=calc_config.get("dispersion"))
    mf = run_scf(mf)
    energy_ha = mf.e_tot

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
