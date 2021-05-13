#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################
# Rebrov PFR reactor script
# Ting-Chen Lee
# Northeastern University
# runs through all reactor conditions
###############################################

import csv
import git
import h5py
import itertools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time

from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct

from rmgpy.molecule import Molecule
from rmgpy.data.base import Database

# unit conversion factors to Cantera's SI
CM = 0.01  # m
MINUTE = 60.0  # s

# dictionary of molecular weights
MOLECULAR_WEIGHTS = {"H": 1.008, "O": 15.999, "C": 12.011, "N": 14.0067}


class PFR:
    """
    A Plug Flow Reactor simulation.
    """

    def __init__(
        self, cantera_filename, dictionary_filename, results_directory, atol, rtol
    ):
        """
        total_length:  reactive length in m
        number_of_reactors:  how many CSTRs
        """
        self.dictionary_filename = dictionary_filename  # species definitions
        self.cantera_filename = cantera_filename
        self.results_directory = results_directory
        self.total_fluxes = {
            "gas": dict(),
            "surf": dict(),
        }  # each dict will have keys like 'C'

        self.load_cantera_file()

        self.atol = atol
        self.rtol = rtol

    def set_geometry(
        self,
        total_length,
        number_of_reactors,
        cat_area_per_gas_volume,
        porosity,
        cross_section_area,
    ):
        self.total_length = total_length  # in m
        self.number_of_reactors = number_of_reactors
        self.cat_area_per_gas_volume = cat_area_per_gas_volume
        self.porosity = porosity
        self.cross_section_area = cross_section_area  # in m2

    def set_initial_conditions(
        self,
        temperature_c,
        feed_mole_fractions,
        surface_coverages,
        mass_flow_rate,
        pressure,
    ):
        self.temperature_c = temperature_c  # in degrees C
        self.feed_mole_fractions = feed_mole_fractions  # mole fractions
        self.mass_flow_rate = mass_flow_rate  # in kg/s
        self.pressure = pressure  # in Pa
        self.surface_coverages = surface_coverages

        self.gas.TPX = temperature_c + 273.15, pressure, feed_mole_fractions
        self.surf.TP = temperature_c + 273.15, pressure
        self.surf.coverages = self.surface_coverages

    def set_parameters(self, **kwargs):
        """
        Set the specified parameters
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            logging.info("Setting %s = %r", key, value)

    @property
    def reactor_length(self):
        return self.total_length / self.number_of_reactors

    @property
    def gas_volume_per_reactor(self):
        return self.cross_section_area * self.reactor_length * self.porosity

    @property
    def cat_area_per_reactor(self):
        return self.cat_area_per_gas_volume * self.gas_volume_per_reactor

    def load_cantera_file(self):
        "Create Gas Solution and Surface Interface from cti file"
        cti_file = self.cantera_filename
        self.gas = ct.Solution(cti_file)
        self.surf = ct.Interface(cti_file, "surface1", [self.gas])

    def save_pictures(self, overwrite=False):
        """
        Save a folder full of molecule pictures, needed for the pretty dot files.

        Saves them in the results directory, in a subfolder "species_pictures".
        Unless you set overwrite=True, it'll leave alone files that are
        already there.
        """
        dictionary_filename = self.dictionary_filename
        specs = Database().get_species(dictionary_filename, resonance=False)

        images_dir = os.path.join(self.results_directory, "species_pictures")
        os.makedirs(images_dir, exist_ok=True)
        for name, species in specs.items():
            filepath = os.path.join(images_dir, name + ".png")
            if not overwrite and os.path.exists(filepath):
                continue
            print(name)
            species.molecule[0].draw(filepath)

    def save_flux_diagrams(self, path="", suffix="", fmt="pdf"):
        """
        Saves the flux diagrams, in the provided path.
        The filenames have a suffix if provided,
        so you can keep them separate and not over-write.
        fmt can be 'pdf' on 'png'
        """
        for element in "CHONX":
            for phase_object in (self.gas, self.surf):
                phase = phase_object.name

                diagram = ct.ReactionPathDiagram(phase_object, element)
                diagram.title = f"Reaction path diagram following {element} in {phase}"
                diagram.label_threshold = 0.01

                dot_file = os.path.join(
                    path,
                    f"reaction_path_{element}_{phase}{'_' if suffix else ''}{suffix}.dot",
                )
                img_file = os.path.join(
                    path,
                    f"reaction_path_{element}_{phase}{'_' if suffix else ''}{suffix}.{fmt}",
                )
                diagram.write_dot(dot_file)
                # print(diagram.get_data())

                print(
                    f"Wrote graphviz input file to '{os.path.join(os.getcwd(), dot_file)}'."
                )

                # Unufortunately this code is duplicated below,
                # so be sure to duplicate any changes you make!
                pretty_dot_file = prettydot(dot_file)
                subprocess.run(
                    [
                        "dot",
                        os.path.abspath(pretty_dot_file),
                        f"-T{fmt}",
                        "-o",
                        os.path.abspath(img_file),
                        "-Gdpi=72",
                    ],
                    cwd=self.results_directory,
                    check=True,
                )
                print(
                    f"Wrote graphviz output file to '{os.path.join(os.getcwd(), img_file)}'."
                )

        # Now do the combined flux
        for name, fluxes_dict in [
            ("mass", self.get_current_fluxes()),
            ("integrated mass", self.total_fluxes),
        ]:

            flux_data_string = self.combine_fluxes(fluxes_dict)
            dot_file = os.path.join(
                path,
                f"reaction_path_{name.replace(' ','_')}{'_' if suffix else ''}{suffix}.dot",
            )
            img_file = os.path.join(
                path,
                f"reaction_path_{name.replace(' ','_')}{'_' if suffix else ''}{suffix}.{fmt}",
            )
            write_flux_dot(
                flux_data_string,
                dot_file,
                threshold=0.01,
                title=f"Reaction path diagram showing combined {name}",
            )
            # Unufortunately this code is duplicated above,
            # so be sure to duplicate any changes you make!
            pretty_dot_file = prettydot(dot_file)

            subprocess.run(
                [
                    "dot",
                    os.path.abspath(pretty_dot_file),
                    f"-T{fmt}",
                    "-o",
                    os.path.abspath(img_file),
                    "-Gdpi=72",
                ],
                cwd=self.results_directory,
                check=True,
            )
            print(
                f"Wrote graphviz output file to '{os.path.abspath(os.path.join(self.results_directory, img_file))}'."
            )

    def show_flux_diagrams(self, suffix="", embed=False):
        """
        Shows the flux diagrams in the notebook.
        Loads them from disk.
        Does not embed them, to keep the .ipynb file small,
        unless embed=True. Use embed=True if you might over-write the files,
        eg. you want to show flux at different points.
        """
        import IPython

        for element in "CHONX":
            for phase_object in (self.gas, self.surf):
                phase = phase_object.name
                img_file = f"reaction_path_{element}_{phase}{'_' if suffix else ''}{suffix}.png"
                display(IPython.display.HTML(f"<hr><h2>{element} {phase}</h2>"))
                if embed:
                    display(
                        IPython.display.Image(filename=img_file, width=400, embed=True)
                    )
                else:
                    display(IPython.display.Image(url=img_file, width=400, embed=False))

            # Now do the combined
            img_file = f"reaction_path_mass{'_' if suffix else ''}{suffix}.png"
            display(IPython.display.HTML(f"<hr><h2>Combined mass</h2>"))
            if embed:
                display(IPython.display.Image(filename=img_file, width=400, embed=True))
            else:
                display(IPython.display.Image(url=img_file, width=400, embed=False))


    def find_species_phase_index(self, species_name):
        """
        Return the phase object (gas or surface) and the index
        of the named species.
        """
        try:
            i = self.gas.species_index(species_name)
            return self.gas, i
        except ValueError:
            i = self.surf.species_index(species_name)
            return self.surf, i

    def change_species_enthalpy(self, species_name, dH):
        """
        Find the species by name and change it's enthlapy by dH (in J/kmol)
        """
        phase, index = self.find_species_phase_index(species_name)
        species = phase.species(index)
        print(f"Initial H(298) = {species.thermo.h(298)/1e6:.1f} kJ/mol")
        dx = dH / ct.gas_constant
        # 'dx' is in fact (delta H / R). Note that R in cantera is 8314.462 J/kmol
        assert isinstance(species.thermo, ct.NasaPoly2)
        # print(species.thermo.coeffs)
        perturbed_coeffs = species.thermo.coeffs.copy()
        perturbed_coeffs[6] += dx
        perturbed_coeffs[13] += dx

        species.thermo = ct.NasaPoly2(
            species.thermo.min_temp,
            species.thermo.max_temp,
            species.thermo.reference_pressure,
            perturbed_coeffs,
        )
        # print(species.thermo.coeffs)
        phase.modify_species(index, species)
        print(f"Modified H(298) = {species.thermo.h(298)/1e6:.1f} kJ/mol")

    def _correct_binding_energy(self, species, delta_atomic_adsoprtion_energies={}, change_rate=False):
        """
        Note: change_rate not yet supported
        Changes the thermo of the provided species, by applying a linear scaling relation
        to correct the adsorption energy.

        :param species: The species to modify (an RMG Species object)
        :param delta_atomic_adsoprtion_energies: a dictionary of changes in atomic adsorption energies to apply.
            mapping for each element an RMG Quantity objects with .value_si giving a value in J/mol.
        :param change_rate: If true, then changes the barrier heights too for all relevant reactions.
        :return: None
        """
        molecule = species.molecule[0]
        # only want/need to do one resonance structure
        surface_sites = []
        for atom in molecule.atoms:
            if atom.is_surface_site():
                surface_sites.append(atom)
        normalized_bonds = {'C': 0., 'O': 0., 'N': 0., 'H': 0.}
        max_bond_order = {'C': 4., 'O': 2., 'N': 3., 'H': 1.}
        for site in surface_sites:
            numbonds = len(site.bonds)
            if numbonds == 0:
                # vanDerWaals
                pass
            else:
                assert len(site.bonds) == 1, "Each surface site can only be bonded to 1 atom"
                bonded_atom = list(site.bonds.keys())[0]
                bond = site.bonds[bonded_atom]
                if bond.is_single():
                    bond_order = 1.
                elif bond.is_double():
                    bond_order = 2.
                elif bond.is_triple():
                    bond_order = 3.
                elif bond.is_quadruple():
                    bond_order = 4.
                else:
                    raise NotImplementedError("Unsupported bond order {0} for binding energy "
                                            "correction.".format(bond.order))

                normalized_bonds[bonded_atom.symbol] += bond_order / max_bond_order[bonded_atom.symbol]


        # now edit the adsorptionThermo using LSR
        change_in_binding_energy = 0.0
        for element in delta_atomic_adsoprtion_energies.keys():
            change_in_binding_energy += delta_atomic_adsoprtion_energies[element].value_si * normalized_bonds[element]
        if change_in_binding_energy != 0.0:
            print(f"Applying LSR correction to {species.label}:")
            if change_rate is True:
                self.change_species_enthalpy_and_rates(species.label, change_in_binding_energy*1000)
            else:
                self.change_species_enthalpy(species.label, change_in_binding_energy*1000)

    def apply_LSRs(self, delta_atomic_adsoprtion_energies):

        self.rmg_spcs = Database().get_species(self.dictionary_filename,resonance=False)

        for species in self.surf.species():
            rmg_spcs = self.rmg_spcs[species.name]
            self._correct_binding_energy(rmg_spcs, delta_atomic_adsoprtion_energies, False)

    def report_rates(self, n=8):
        """
        Report the highest n reaction rates of progress in each phase,
        net, forward, and reverse.
        """
        gas, surf = self.gas, self.surf
        cat_area_per_vol = self.cat_area_per_gas_volume

        print("\nHighest net rates of progress, gas")
        for i in np.argsort(abs(gas.net_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {gas.reaction_equation(i):48s}  {gas.net_rates_of_progress[i]:8.1g}"
            )
        print("\nHighest net rates of progress, surface")
        for i in np.argsort(abs(surf.net_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {surf.reaction_equation(i):48s}  {cat_area_per_vol*surf.net_rates_of_progress[i]:8.1g}"
            )
        print("\nHighest forward rates of progress, gas")
        for i in np.argsort(abs(gas.forward_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {gas.reaction_equation(i):48s}  {gas.forward_rates_of_progress[i]:8.1g}"
            )
        print("\nHighest forward rates of progress, surface")
        for i in np.argsort(abs(surf.forward_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {surf.reaction_equation(i):48s}  {cat_area_per_vol*surf.forward_rates_of_progress[i]:8.1g}"
            )
        print("\nHighest reverse rates of progress, gas")
        for i in np.argsort(abs(gas.reverse_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {gas.reaction_equation(i):48s}  {gas.reverse_rates_of_progress[i]:8.1g}"
            )
        print("\nHighest reverse rates of progress, surface")
        for i in np.argsort(abs(surf.reverse_rates_of_progress))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {surf.reaction_equation(i):48s}  {cat_area_per_vol*surf.reverse_rates_of_progress[i]:8.1g}"
            )

        print(
            f"\nSurface rates have been scaled by surface/volume ratio {cat_area_per_vol:.1e} m2/m3"
        )
        print("So are on a similar basis of volume of gas")
        print(" kmol / m3 / s")

    def report_rate_constants(self, n=8):
        """
        Report the highest n reaction rate coefficients in each phase,
        forward, and reverse.
        """
        gas, surf = self.gas, self.surf

        print("\nHighest forward rate constants, gas")
        for i in np.argsort(abs(gas.forward_rate_constants))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {gas.reaction_equation(i):48s}  {gas.forward_rate_constants[i]:8.1e}"
            )
        print("\nHighest forward rate constants, surface")
        for i in np.argsort(abs(surf.forward_rate_constants))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {surf.reaction_equation(i):48s}  {surf.forward_rate_constants[i]:8.1e}"
            )
        print("\nHighest reverse rate constants, gas")
        for i in np.argsort(abs(gas.reverse_rate_constants))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {gas.reaction_equation(i):48s}  {gas.reverse_rate_constants[i]:8.1e}"
            )
        print("\nHighest reverse rate constants, surface")
        for i in np.argsort(abs(surf.reverse_rate_constants))[-1:-n:-1]:
            # top n in descending order
            print(
                f"{i:3d} : {surf.reaction_equation(i):48s}  {surf.reverse_rate_constants[i]:8.1e}"
            )

        print(
            "Units are a combination of kmol, m^3 and s, that depend on the rate expression for the reaction."
        )

    def get_steady_state_starting_coverages(
        self,
        temperature_c,
        pressure,
        feed_mole_fractions,
        gas_volume_per_reactor,
        cat_area_per_gas_volume,
    ):
        """
        To find the starting coverages, we run the gas to equilibrium,
        (i.e mostly burned products)  then put that in steady state
        with the surface.

        May not be working
        """
        gas, surf = self.gas, self.surf
        gas.TPX = temperature_c + 273.15, pressure, feed_mole_fractions
        TPY = gas.TPY  # store to restore
        # gas.equilibrate('TP')
        r = ct.IdealGasReactor(gas, energy="off")
        r.volume = gas_volume_per_reactor
        cat_area_per_reactor = cat_area_per_gas_volume * gas_volume_per_reactor
        rsurf = ct.ReactorSurface(surf, r, A=cat_area_per_reactor)
        sim = ct.ReactorNet([r])
        sim.advance(1e-3)
        surf()
        starting_coverages = surf.coverages

        gas.TPY = TPY  # restore to starting conditions
        del (r, rsurf)
        return starting_coverages

    def get_current_fluxes(self):
        """
        Get all the current fluxes.
        Returns a dict like:
         `fluxes['gas']['C'] = ct.ReactionPathDiagram(self.gas, 'C')` etc.
        """
        fluxes = {"gas": dict(), "surf": dict()}
        for element in "HOCN":
            fluxes["gas"][element] = ct.ReactionPathDiagram(self.gas, element)
            fluxes["surf"][element] = ct.ReactionPathDiagram(self.surf, element)
        element = "X"  # just the surface
        fluxes["surf"][element] = ct.ReactionPathDiagram(self.surf, element)
        return fluxes

    def add_fluxes(self):
        """
        Add the current fluxes to the stored totals.
        """
        gas_fluxes = self.total_fluxes["gas"]
        surf_fluxes = self.total_fluxes["surf"]
        for element in "HOCN":
            try:
                gas_fluxes[element].add(ct.ReactionPathDiagram(self.gas, element))
            except KeyError:
                gas_fluxes[element] = ct.ReactionPathDiagram(self.gas, element)
            try:
                surf_fluxes[element].add(ct.ReactionPathDiagram(self.surf, element))
            except KeyError:
                surf_fluxes[element] = ct.ReactionPathDiagram(self.surf, element)
        # Now do the 'X' for just the surface
        element = "X"
        try:
            surf_fluxes[element].add(ct.ReactionPathDiagram(self.surf, element))
        except KeyError:
            surf_fluxes[element] = ct.ReactionPathDiagram(self.surf, element)

    def save_flux_data(self, filename_stem=None):
        """
        Save the current and integrated fluxes.
        Also returns them.
        """
        fluxes = {
            "current": self.get_current_fluxes(),
            "integrated": self.total_fluxes,
        }
        flux_strings = dict()
        for flux_type, d1 in fluxes.items():
            flux_strings[flux_type] = dict()
            for phase, d2 in d1.items():
                flux_strings[flux_type][phase] = dict()
                for element, flux in d2.items():
                    flux_strings[flux_type][phase][element] = flux.get_data()
            flux_strings[flux_type]["combined"] = dict()
            flux_strings[flux_type]["combined"]["mass"] = self.combine_fluxes(d1)

        if filename_stem:
            path = os.path.join(self.results_directory, filename_stem + ".json")
            with open(path, "w") as f:
                json.dump(flux_strings, f)
        return flux_strings

    def combine_fluxes(self, fluxes_dict):
        """
        Combined a dict of dicts of flux diagrams into one.

        Fluxes should be a dict with entries like
           fluxes['gas']['C'] = ct.ReactionPathDiagram(self.gas, 'C')

        Returns the flux diagram a string in the format you'd get from
        ct.ReactionPathdiagram.get_data()
        """
        # getting the entire net rates of the system
        temp_flux_data = dict()
        species = set()
        for element in "HOCN":
            for phase in ("gas", "surf"):
                data = fluxes_dict[phase][element].get_data().strip().splitlines()
                if not data:
                    # eg. if there's no gas-phase reactions involving C
                    continue
                species.update(data[0].split())  # First line is a list of species
                for line in data[1:]:  # skip the first line

                    s1, s2, fwd, rev = line.split()
                    these_fluxes = np.array([float(fwd), float(rev)])

                    if all(these_fluxes == 0):
                        continue

                    # multiply by atomic mass of the element
                    these_fluxes *= MOLECULAR_WEIGHTS[element]

                    # for surface reactions, multiply by the catalyst area per volume in a reactor
                    if phase == "surf":
                        these_fluxes *= self.cat_area_per_gas_volume

                    try:
                        # Try adding in this direction
                        temp_flux_data[(s1, s2)] += these_fluxes
                    except KeyError:
                        try:
                            # Try adding in reverse direction
                            temp_flux_data[(s2, s1)] -= these_fluxes
                        except KeyError:
                            # Neither direction there yet, so create in this direction
                            temp_flux_data[(s1, s2)] = these_fluxes

        output = " ".join(species) + "\n"
        output += "\n".join(
            f"{s1} {s2} {fwd} {rev}" for (s1, s2), (fwd, rev) in temp_flux_data.items()
        )
        return output

    def plug_flow_simulation(self, output_filename, energy='off'):
        """
        PLUG FLOW REACTOR SIMULATION

        The plug flow reactor is represented by a linear chain of zero-dimensional
        reactors. The gas at the inlet to the first one has the specified inlet
        composition, and for all others the inlet composition is fixed at the
        composition of thereactor immediately upstream. Since in a PFR model there
        is no diffusion, the upstream reactors are not affected by any downstream
        reactors, and therefore the problem may be solved by simply marching from
        the first to last reactor, integrating each one to steady state.
        """

        gas, surf = self.gas, self.surf

        gas.TPX = self.temperature_c + 273.15, self.pressure, self.feed_mole_fractions
        surf.TP = self.temperature_c + 273.15, self.pressure
        surf.coverages = self.surface_coverages

        TDY = gas.TDY

        # create a new reactor
        gas.TDY = TDY
        r = ct.IdealGasReactor(gas, energy=energy)
        r.volume = self.gas_volume_per_reactor

        # create a reservoir to represent the reactor immediately upstream. Note
        # that the gas object is set already to the state of the upstream reactor
        upstream = ct.Reservoir(gas, name="upstream")

        # create a reservoir for the reactor to exhaust into. The composition of
        # this reservoir is irrelevant.
        downstream = ct.Reservoir(gas, name="downstream")

        # Add the reacting surface to the reactor. The area is set to the desired
        # catalyst area in the reactor.
        rsurf = ct.ReactorSurface(surf, r, A=self.cat_area_per_reactor)

        # The mass flow rate into the reactor will be fixed by using a
        # MassFlowController object.
        m = ct.MassFlowController(upstream, r, mdot=self.mass_flow_rate)

        # We need an outlet to the downstream reservoir. This will determine the
        # pressure in the reactor. The value of K will only affect the transient
        # pressure difference.
        v = ct.PressureController(r, downstream, master=m, K=1e-5)

        sim = ct.ReactorNet([r])
        sim.max_err_test_fails = 24

        # set relative and absolute tolerances on the simulation
        sim.rtol = self.rtol
        sim.atol = self.atol

        sim.verbose = False
        r.volume = self.gas_volume_per_reactor

        os.makedirs(self.results_directory, exist_ok=True)
        outfile = open(os.path.join(self.results_directory, output_filename), "w")
        writer = csv.writer(outfile)
        writer.writerow(
            ["Distance (mm)", "T (C)", "P (atm)"]
            + gas.species_names
            + surf.species_names
            + ["gas_heat", "surface_heat", "alpha"]
        )

        print(
            "    distance(mm)    T(C)    NH3(6)    O2(2)    N2(4)    N2O(7)    NO(5)    H2O(3)    alpha"
        )

        for n in range(self.number_of_reactors):

            # Set the state of the reservoir to match that of the previous reactor
            gas.TDY = TDY = r.thermo.TDY
            upstream.syncState()
            sim.reinitialize()
            previous_coverages = surf.coverages  # in case we want to retry

            if n > 0:  # Add a first row in the CSV with just the feed
                try:
                    sim.advance_to_steady_state()
                except ct.CanteraError:
                    t = sim.time
                    sim.set_initial_time(0)
                    gas.TDY = TDY
                    surf.coverages = previous_coverages
                    r.syncState()
                    sim.reinitialize()
                    new_target_time = 0.01 * t
                    logging.warning(
                        f"Couldn't reach {t:.1g} s so going to try {new_target_time:.1g} s"
                    )
                    # self.save_flux_diagrams(path=f"data/H3NX(29)+{delta}")
                    # self.show_flux_diagrams(embed=True)
                    # self.report_rates()
                    # self.report_rate_constants()
                    try:
                        sim.advance(new_target_time)
                    except ct.CanteraError:
                        outfile.close()
                        raise

                # dont add fluxes at distance=0 because you just
                # have an almost infinite flux onto vacant surface
                self.add_fluxes()  # for the integration

            distance_mm = n * self.reactor_length * 1.0e3  # distance in mm

            # heat evolved by gas phase reaction:
            gas_heat = surface_heat = alpha = 1
            # heat evolved by surf phase reaction:
            surface_heat = self.cat_area_per_gas_volume * np.dot(
                surf.net_rates_of_progress, surf.delta_enthalpy
            )
            # fraction of heat release that is on surface:
            alpha = surface_heat / (surface_heat + gas_heat)

            if not n % 10:
                print(
                    "    {:10f}  {:7.1f} {:10f} {:10f} {:10f}  {:10f}  {:10f}  {:10f}  {:5.1e}".format(
                        distance_mm,
                        r.T - 273.15,
                        *gas[
                            "NH3(6)", "O2(2)", "N2(4)", "N2O(7)", "NO(5)", "H2O(3)"
                        ].X,
                        alpha,
                    )
                )
                print("Highest surface coverages are:")
                for i in np.argsort(surf.coverages)[::-1][:5]:
                    print(surf.species_name(i), round(surf.coverages[i], 4))

            if n in (1, int(self.number_of_reactors / 2), self.number_of_reactors - 1):
                # Will save at start, midpoint, and end
                self.save_flux_diagrams(
                    path=self.results_directory,
                    suffix=f"Temp_{r.T-273.15:.0f}C_Dist_{distance_mm:.1f}",
                )

            if (not (n-1) % 100) or n == (self.number_of_reactors - 1):
                self.save_flux_data(
                    f"flux_data_Temp_{r.T-273.15:.0f}C_Dist_{distance_mm:.1f}"
                )

            # write the gas mole fractions and surface coverages vs. distance
            writer.writerow(
                [distance_mm, r.T - 273.15, r.thermo.P / ct.one_atm]
                + list(gas.X)
                + list(surf.coverages)
                + [gas_heat, surface_heat, alpha]
            )

        outfile.close()
        print("Results saved to '{0}'".format(output_filename))

    def change_some_rates(self, save_multipliers=True):
        """
        For experimenting with the mechanism, this function lets you change
        some reaction rates.
        """
        surf = self.surf
        multiplier = 1000
        # for i in [71,248,270,139,73,44,48,71,48,22]:
        #     print(surf.reaction_equation(i))
        #     surf.set_multiplier(multiplier,i)
        # for i in [65,32,70,12]:
        for i in [82, 103, 84, 99, 95, 259, 283]:
            print(surf.reaction_equation(i))
            surf.set_multiplier(1 / multiplier, i)

        for rxn in surf.reactions():
            if rxn.is_sticking_coefficient:
                set_max_sticking_coeff(rxn, A=1.0)

        if save_multipliers:
            with open(
                os.path.join(self.results_directory, "surf_multipliers.csv"), "w"
            ) as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["index", "reaction", "multiplier"])
                for i, rxn in enumerate(surf.reactions()):
                    print(i, rxn.equation, surf.multiplier(i))
                    csv_writer.writerow([i, rxn.equation, surf.multiplier(i)])

    def modify_mechanism(self):
        """
        Whatever changes we want to make to the chemistry after loading it
        """

        self.change_some_rates()

        fix_rates(self.gas, 1e18)
        fix_rates(self.surf, 1e21)


def fix_rates(phase, limit):
    """
    Fix reverse reaction rates that are too fast.
    """
    for i in np.argsort(abs(phase.reverse_rate_constants))[-1:0:-1]:
        if phase.reverse_rate_constants[i] < limit:
            break
        print(
            f"Before: {i:3d} : {phase.reaction_equation(i):48s}  {phase.reverse_rate_constants[i]:8.1e}"
        )
        multiplier = limit / phase.reverse_rate_constants[i]
        phase.set_multiplier(multiplier, i)
        print(
            f"After:  {i:3d} : {phase.reaction_equation(i):48s}  {phase.reverse_rate_constants[i]:8.1e}"
        )


def set_max_sticking_coeff(rxn, A=0.75):
    """
    Fix the maximum sticking coefficient for the given reaction.

    ToDo: currently this ignores n and Ea in an Arrhenius expression
    and replaces with a uniform A value..
    Also, doesn't yet actually update the kinetics object in Cantera memory.
    """
    assert rxn.is_sticking_coefficient
    if rxn.rate.pre_exponential_factor > A:
        old_stick = rxn.rate
        rate = ct.Arrhenius(A)
        rxn.rate = rate
        print(
            f"Changed sticking coeff for {rxn.equation} from {old_stick!r} to {rxn.rate.pre_exponential_factor}"
        )
        raise NotImplementedError("Haven't updated the kinetics object in Cantera")


def write_flux_dot(flux_data_string, out_file_path, threshold=0.01, title=""):
    """
    Takes a flux data string fromatted as from ct.ReactionPathdiagram.get_data()
    (or from combine_fluxes) and makes a graphviz .dot file.

    Fluxes below 'threshold' are not plotted.
    """

    output = ["digraph reaction_paths {", "center=1;"]

    flux_data = {}
    species_dict = {}
    flux_data_lines = flux_data_string.splitlines()
    species = flux_data_lines[0].split()

    for line in flux_data_lines[1:]:
        s1, s2, fwd, rev = line.split()
        net = float(fwd) + float(rev)
        if net < 0.0:  # if net is negative, switch s1 and s2 so it is positive
            flux_data[(s2, s1)] = -1 * net
        else:
            flux_data[(s1, s2)] = net

        # renaming species to dot compatible names
        if s1 not in species_dict:
            species_dict[s1] = "s" + str(len(species_dict) + 1)
        if s2 not in species_dict:
            species_dict[s2] = "s" + str(len(species_dict) + 1)

    # getting the arrow widths
    largest_rate = max(flux_data.values())

    added_species = {}  # dictionary of species that show up on the diagram
    for (s1, s2), net in flux_data.items():  # writing the node connections

        flux_ratio = net / largest_rate
        if abs(flux_ratio) < threshold:
            continue  # don't include the paths that are below the threshold

        pen_width = (
            1.0 - 4.0 * np.log10(flux_ratio / threshold) / np.log10(threshold) + 1.0
        )
        # pen_width = ((net - smallest_rate) / (largest_rate - smallest_rate)) * 4 + 2
        arrow_size = min(6.0, 0.5 * pen_width)
        output.append(
            f'{species_dict[s1]} -> {species_dict[s2]} [fontname="Helvetica", penwidth={pen_width:.2f}, arrowsize={arrow_size:.2f}, color="0.7, {flux_ratio+0.5:.3f}, 0.9", label="{flux_ratio:0.3g}"];'
        )

        added_species[s1] = species_dict[s1]
        added_species[s2] = species_dict[s2]

    for (
        species,
        s_index,
    ) in added_species.items():  # writing the species translations
        output.append(f'{s_index} [ fontname="Helvetica", label="{species}"];')

    title_string = (r"\l " + title) if title else ""
    output.append(f' label = "Scale = {largest_rate}{title_string}";')
    output.append(' fontname = "Helvetica";')
    output.append("}\n")

    directory = os.path.split(out_file_path)[0]
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_file_path, "w") as out_file:
        out_file.write("\n".join(output))
    return "\n".join(output)


def prettydot(dotfilepath, strip_line_labels=False):
    """
    Make a prettier version of the dot file (flux diagram)

    Assumes the species pictures are stored in a directory
    called 'species_pictures' alongside the dot file.
    """
    import os, sys, re
    import subprocess

    pictures_directory = os.path.join(os.path.split(dotfilepath)[0], "species_pictures")

    if strip_line_labels:
        print("stripping edge (line) labels")

    # replace this:
    #  s10 [ fontname="Helvetica", label="C11H23J"];
    # with this:
    #  s10 [ shapefile="mols/C11H23J.png" label="" width="1" height="1" imagescale=true fixedsize=true color="white" ];

    reSize = re.compile('size="5,6"\;page="5,6"')
    reNode = re.compile(
        '(?P<node>s\d+)\ \[\ fontname="Helvetica",\ label="(?P<label>[^"]*)"\]\;'
    )

    rePicture = re.compile("(?P<smiles>.+?)\((?P<id>\d+)\)\.png")
    reLabel = re.compile("(?P<name>.+?)\((?P<id>\d+)\)$")

    species_pictures = dict()
    for picturefile in os.listdir(pictures_directory):
        match = rePicture.match(picturefile)
        if match:
            species_pictures[match.group("id")] = picturefile
        else:
            pass
            # print(picturefile, "didn't look like a picture")

    filepath = dotfilepath

    if not open(filepath).readline().startswith("digraph"):
        raise ValueError("{0} - not a digraph".format(filepath))

    infile = open(filepath)
    prettypath = filepath + "-pretty"
    outfile = open(prettypath, "w")

    for line in infile:
        (line, changed_size) = reSize.subn('size="12,12";page="12,12"', line)
        match = reNode.search(line)
        if match:
            label = match.group("label")
            idmatch = reLabel.match(label)
            if idmatch:
                idnumber = idmatch.group("id")
                if idnumber in species_pictures:
                    line = (
                        '%s [ image="species_pictures/%s" label="" width="0.5" height="0.5" imagescale=false fixedsize=false color="none" ];\n'
                        % (match.group("node"), species_pictures[idnumber])
                    )

        # rankdir="LR" to make graph go left>right instead of top>bottom

        if strip_line_labels:
            line = re.sub('label\s*=\s*"\s*[\d.]+"', 'label=""', line)

        # change colours
        line = re.sub('color="0.7,\ (.*?),\ 0.9"', r'color="1.0, \1, 0.7*\1"', line)

        outfile.write(line)

    outfile.close()
    infile.close()
    print(f"Graph saved to: {prettypath}")
    return prettypath


def run(reduced=False, nh3x_delta=10):
    """
    Run a simulation of the experimental packed bed reactor.

    nh3x_delta in kcal/mol
    """

    if reduced:
        cantera_filename = f"data/NH3X{nh3x_delta:+g}-rx5/chem_annotated_reduced_{reduced}.cti"
        raise NotImplementedError("Not working for now")
    else:
        cantera_filename = f"chem_annotated.cti"

    # different temperature for each SLURM job
    cat_area_per_vol_options = [3e3, 3e4, 3e5, 3e6, 3e7, 3e8, 3e9] # m2/m3
    temperature_c_options = [100, 135, 170, 205, 240, 275, 310, 345, 380, 415, 450, 485] # ºC

    settings  = list(itertools.product(cat_area_per_vol_options,
                                    temperature_c_options,
                                    ))

    array_i = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))

    cat_area_per_gas_volume, temperature_c = settings[array_i - 1]

    results_directory = f"PFR_results/A{cat_area_per_gas_volume:.0f}"

    cantera_full_path = os.path.join(results_directory, cantera_filename)
    # If the cantera file isn't in the results directory, copy it there
    if not os.path.exists(cantera_full_path):
        os.makedirs(results_directory, exist_ok=True)
        shutil.copy2(cantera_filename, results_directory)
        cantera_full_path = os.path.join(results_directory, os.path.split(cantera_filename)[-1])
    # If it is already in the results directory, use it
    cantera_filename = cantera_full_path

    rtol = 1e-10
    atol = 1e-20

    pfr = PFR(
        cantera_filename=cantera_filename,
        dictionary_filename="species_dictionary.txt",
        results_directory=results_directory,
        atol=atol,
        rtol=rtol,
    )

    pfr.save_pictures()

    if nh3x_delta:
        delta_j_kmol = nh3x_delta * 4.184e6  # j/kmol
        pfr.change_species_enthalpy("H3NX(29)", delta_j_kmol)

    # change_some_rates(surf, results_directory, save_multipliers=(os.getenv("SLURM_ARRAY_TASK_ID","1")=="1"))

    # Feed mol fractions with NH2OH and HNO3
    # feed_mole_fractions = {
    #     'N2O(7)': 0.32,
    #     'NO(5)': 0.35,
    #     'H2O(3)': 0.21,
    #     'He': 0.09,
    # }

    # Feed mol fractions with NH3
    feed_mole_fractions = {
        "NH3(6)": 0.011,  # ammonia
        "O2(2)": 0.88,  # oxygen
        "H2O(3)": 0.001,  # water
        "He": 0.108,  # Helium
    }

    #######################################################################
    # Input Parameters for combustor
    #######################################################################

    # The PFR will be simulated by a chain of 'number_of_reactors' stirred reactors.
    number_of_reactors = 1100
    total_length = 0.9 * CM  # Catalyst bed length. 9mm
    cross_section_area = np.pi * (0.014 * CM) ** 2 /2 # Catalyst bed area. 280µm diameter.
    porosity = 0.38
    cat_area_per_gas_volume = 2.8571428e4  # m2/m3
    print(
        f"\nCatalyst area per volume in use for this simulation: {cat_area_per_gas_volume :.2e} m2/m3"
    )
    pfr.set_geometry(
        total_length,
        number_of_reactors,
        cat_area_per_gas_volume,
        porosity,
        cross_section_area,
    )

    temperature_c = temperature_c  # Initial Temperature in Celsius
    mass_flow_rate = 5.74931156e-10  # kg/s
    # temperature_c = 750.1092476 - 273.15
    print(f"Initial temperature {temperature_c :.1f} ºC")
    pressure = ct.one_atm  # constant

    # surface_coverages = get_steady_state_starting_coverages(
    #     gas, surf, temperature_c, pressure, feed_mole_fractions, gas_volume_per_reactor, cat_area_per_gas_volume
    # )
    surface_coverages = "X(1): 1.0"

    pfr.set_initial_conditions(
        temperature_c, feed_mole_fractions, surface_coverages, mass_flow_rate, pressure
    )

    # specifty outfile to write results of simulation
    output_filename = f"surf_pfr_isothermal_Pt_Temp_{temperature_c}.csv"
    pfr.plug_flow_simulation(output_filename)

    return pfr

if __name__ == "__main__":
    pfr = run(reduced=False, nh3x_delta=10)