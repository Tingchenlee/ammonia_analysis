###############################################
# Rebrov reactor batch script
# Ting-Chen Lee
# Northeastern University
# runs through all reactor conditions
###############################################

import pandas as pd
import numpy as np
import time
import cantera as ct
from matplotlib import pyplot as plt
import csv
import math
import os
import sys
import re
import itertools
import logging
from collections import defaultdict
import git


# todo:
# add in save_pictures so we save it in the proper folder


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


def show_flux_diagrams(self, suffix="", embed=False):
    """
    Shows the flux diagrams in the notebook.
    Loads them from disk.
    Does not embed them, to keep the .ipynb file small,
    unless embed=True. Use embed=True if you might over-write the files,
    eg. you want to show flux at different points.
    """
    import IPython

    for element in "CHONX": #try NH3?
        for phase_object in (self.gas, self.surf):
            phase = phase_object.name
            img_file = (
                f"reaction_path_{element}_{phase}{'_' if suffix else ''}{suffix}.png"
            )
            display(IPython.display.HTML(f"<hr><h2>{element} {phase}</h2>"))
            if embed:
                display(IPython.display.Image(filename=img_file, width=400, embed=True))
            else:
                display(IPython.display.Image(url=img_file, width=400, embed=False))

        # Now do the combined
        img_file = f"reaction_path_mass{'_' if suffix else ''}{suffix}.png"
        display(IPython.display.HTML(f"<hr><h2>Combined mass</h2>"))
        if embed:
            display(IPython.display.Image(filename=img_file, width=400, embed=True))
        else:
            display(IPython.display.Image(url=img_file, width=400, embed=False))


def save_flux_diagrams(*phases, suffix="", timepoint=""):
    import pandas as pd
    import numpy as np
    import time
    import cantera as ct
    from matplotlib import pyplot as plt
    import csv
    import math
    import os
    import sys
    import re
    import itertools
    import logging
    from collections import defaultdict

    """
    Saves the flux diagrams. The filenames have a suffix if provided,
    so you can keep them separate and not over-write.
    """
    for element in "CHONX":  #try NH3?
        for phase_object in phases:
            phase = phase_object.name

            diagram = ct.ReactionPathDiagram(phase_object, element)
            diagram.title = f"Reaction path diagram following {element} in {phase}"
            diagram.label_threshold = 0.001

            dot_file = f"{suffix}/reaction_path_{element}_{phase}_{timepoint}.dot"
            img_file = f"{suffix}/reaction_path_{element}_{phase}_{timepoint}.png"
            dot_bin_path = (
                "/Users/lee.ting/Code/anaconda3/pkgs/graphviz-2.40.1-hefbbd9a_2/bin/dot" 
                #maybe try "/home/lee.ting/.conda/pkgs/graphviz-2.40.1-h21bd128_2/bin/dot"
            )
            img_path = os.path.join(os.getcwd(), img_file)
            diagram.write_dot(dot_file)
            # print(diagram.get_data())

            print(
                f"Wrote graphviz input file to '{os.path.join(os.getcwd(), dot_file)}'."
            )
            os.system(f"dot {dot_file} -Tpng -o{img_file} -Gdpi=200")
            print(f"Wrote graphviz output file to '{img_path}'.")


def run_reactor(
    cti_file,
    t_array=[548],
    p_array=[1],
    v_array=[2.7155e-8], #14*7*(140e-4)^2*π/2*0.9=0.02715467 (cm3)
    o2_array=[0.88],
    nh3_array=[0.066],
    rtol=1.0e-11,
    atol=1.0e-22,
    reactor_type=0,
    energy="off",
    sensitivity=False,
    sensatol=1e-6,
    sensrtol=1e-6,
    reactime=1e5,
):
#14 aluminum plates, each of them containing seven semi-cylindrical mi-crochannels of 280 µm width 
# and 140 µm depth, 9 mm long, arranged at equal distances of 280 µm 

    import pandas as pd
    import numpy as np
    import time
    import cantera as ct
    from matplotlib import pyplot as plt
    import csv
    import math
    import os
    import sys
    import re
    import itertools
    import logging
    from collections import defaultdict
    import git

    try:
        array_i = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    except TypeError:
        array_i = 0

    # get git commit hash and message

    repo = git.Repo("/work/westgroup/lee.ting/cantera/ammonia/")
    git_sha = str(repo.head.commit)[0:6]
    git_msg = str(repo.head.commit.message)[0:20].replace(" ", "_").replace("'", "_")

    # this should probably be outside of function
    settings = list(itertools.product(t_array, p_array, v_array, o2_array, nh3_array))

    # constants
    pi = math.pi

    # set initial temps, pressures, concentrations
    temp = settings[array_i][0]  # kelvin
    temp_str = str(temp)[0:3]
    pressure = settings[array_i][1] * ct.one_atm  # Pascals

    X_o2 = settings[array_i][3]
    x_O2_str = str(X_o2)[0:3].replace(".", "_")
    if X_o2 == 0.88:
        X_he = 0.054
    else:
        X_he = 0
    X_nh3 = (1 - (X_o2 + X_he)) * (settings[array_i][4])
    x_NH3_str = str(X_nh3)[0:8].replace(".", "_")
    
    mw_nh3 = 17.0306e-3  # [kg/mol]
    mw_o2 = 31.999e-3  # [kg/mol]
    mw_he = 4.002602e-3  # [kg/mol]

    o2_ratio = X_nh3 / X_o2

    # CO/CO2/H2/H2: typical is
    concentrations_rmg = {"O2(2)": X_o2, "NH3(6)": X_nh3, "He": X_he}

    # initialize cantera gas and surface
    gas = ct.Solution(cti_file, "gas")

    # surf_grab = ct.Interface(cti_file,'surface1_grab', [gas_grab])
    surf = ct.Interface(cti_file, "surface1", [gas])

    # gas_grab.TPX =
    gas.TPX = temp, pressure, concentrations_rmg
    surf.TP = temp, pressure

    # create gas inlet
    inlet = ct.Reservoir(gas)

    # create gas outlet
    exhaust = ct.Reservoir(gas)

    # Reactor volume
    rradius = 1.4e-4 #140µm to 0.00014m
    rlength = 9e-3 #9mm to 0.009m
    rvol = (rradius ** 2) * pi * rlength / 2

    # Catalyst Surface Area
    site_density = (surf.site_density * 1000)  # [mol/m^2]cantera uses kmol/m^2, convert to mol/m^2
    cat_area = rradius * 2 / 2 * pi * rlength # [m^3]
    #suface site density = 1.86e-9 mol/cm2 = 1.96e-5 mol/m2; molecular weight for Pt = 195.084 g/mol
    # per kg has 5.125997 moles Pt = 5.125997*6.022e23/1.12e15(cm-2) = 2.756138744e9 cm2/kg = 2.756e5m2/kg

    # reactor initialization
    if reactor_type == 0:
        r = ct.Reactor(gas, energy=energy)
        reactor_type_str = "Reactor"
    elif reactor_type == 1:
        r = ct.IdealGasReactor(gas, energy=energy)
        reactor_type_str = "IdealGasReactor"
    elif reactor_type == 2:
        r = ct.ConstPressureReactor(gas, energy=energy)
        reactor_type_str = "ConstPressureReactor"
    elif reactor_type == 3:
        r = ct.IdealGasConstPressureReactor(gas, energy=energy)
        reactor_type_str = "IdealGasConstPressureReactor"

    rsurf = ct.ReactorSurface(surf, r, A=cat_area)
    r.volume = rvol
    surf.coverages = "X(1):1.0"

    # flow controllers (Graaf measured flow at 293.15 and 1 atm)
    one_atm = ct.one_atm
    FC_temp = 293.15
    volume_flow = settings[array_i][2]  # [m^3/s]
    molar_flow = volume_flow * one_atm / (8.3145 * FC_temp)  # [mol/s]
    mass_flow = molar_flow * (X_nh3 * mw_nh3 + X_o2 * mw_o2 + X_he * mw_he)  # [kg/s]
    mfc = ct.MassFlowController(inlet, r, mdot=mass_flow)

    # A PressureController has a baseline mass flow rate matching the 'master'
    # MassFlowController, with an additional pressure-dependent term. By explicitly
    # including the upstream mass flow rate, the pressure is kept constant without
    # needing to use a large value for 'K', which can introduce undesired stiffness.
    outlet_mfc = ct.PressureController(r, exhaust, master=mfc, K=0.01)

    # initialize reactor network
    sim = ct.ReactorNet([r])

    # set relative and absolute tolerances on the simulation
    sim.rtol = 1.0e-11
    sim.atol = 1.0e-22

    #################################################
    # Run single reactor
    #################################################

    # round numbers so they're easier to read
    # temp_str = '%s' % '%.3g' % tempn

    cat_area_str = "%s" % "%.3g" % cat_area
    results_path = (
        os.path.dirname(os.path.abspath(__file__))
        + f"/{git_sha}_{git_msg}/{reactor_type_str}/transient/{temp_str}/results"
    )
    # results_path_csp = (
    #     os.path.dirname(os.path.abspath(__file__))
    #     + f"/{git_sha}_{git_msg}/{reactor_type_str}/transient/{temp_str}/results/csp"
    # )
    flux_path = (
        os.path.dirname(os.path.abspath(__file__))
        + f"/{git_sha}_{git_msg}/{reactor_type_str}/transient/{temp_str}/flux_diagrams/{x_O2_str}/{x_NH3_str}"
    )
    try:
        os.makedirs(results_path, exist_ok=True)
    except OSError as error:
        print(error)

    # try:
    #     os.makedirs(results_path_csp, exist_ok=True)
    # except OSError as error:
    #     print(error)

    try:
        os.makedirs(flux_path, exist_ok=True)
    except OSError as error:
        print(error)

    gas_ROP_str = [i + " ROP [kmol/m^3 s]" for i in gas.species_names]

    # surface ROP reports gas and surface ROP. these values might be redundant, not sure.

    gas_surf_ROP_str = [i + " surface ROP [kmol/m^2 s]" for i in gas.species_names]
    surf_ROP_str = [i + " ROP [kmol/m^2 s]" for i in surf.species_names]

    gasrxn_ROP_str = [i + " ROP [kmol/m^3 s]" for i in gas.reaction_equations()]
    surfrxn_ROP_str = [i + " ROP [kmol/m^2 s]" for i in surf.reaction_equations()]

    output_filename = (
        results_path
        + f"/Spinning_basket_area_{cat_area_str}_energy_{energy}"
        + f"_temp_{temp}_O2_{x_O2_str}_NH3_{x_NH3_str}.csv"
    )
    # output_filename_csp = (
    #     results_path_csp
    #     + f"/Spinning_basket_area_{cat_area_str}_energy_{energy}"
    #     + f"_temp_{temp}_h2_{x_o2_str}_COCO2_{x_CO_CO2_str}.csv"
    # )
    outfile = open(output_filename, "w")
    # outfile_csp = open(output_filename_csp, "w")
    writer = csv.writer(outfile)
    # writer_csp = csv.writer(outfile_csp)

    # Sensitivity atol, rtol, and strings for gas and surface reactions if selected
    # slows down script by a lot
    if sensitivity:
        sim.rtol_sensitivity = sensrtol
        sim.atol_sensitivity = sensatol
        sens_species = ["NH3(6)"]  #change THIS to your species, can add "," and other species

        # turn on sensitive reactions/species
        for i in range(gas.n_reactions):
            r.add_sensitivity_reaction(i)

        for i in range(surf.n_reactions):
            rsurf.add_sensitivity_reaction(i)

        # for i in range(gas.n_species):
        #     r.add_sensitivity_species_enthalpy(i)

        # for i in range(surf.n_species):
        #     rsurf.add_sensitivity_species_enthalpy(i)

        for j in sens_species:
            gasrxn_sens_str = [
                j + " sensitivity to " + i for i in gas.reaction_equations()
            ]
            surfrxn_sens_str = [
                j + " sensitivity to " + i for i in surf.reaction_equations()
            ]
            # gastherm_sens_str = [j + " thermo sensitivity to " + i for i in gas.species_names]
            # surftherm_sens_str = [j + " thermo sensitivity to " + i for i in surf.species_names]
            sens_list = gasrxn_sens_str + surfrxn_sens_str  # + gastherm_sens_str

        writer.writerow(
            [
                "T (C)",
                "P (atm)",
                "V (M^3/s)",
                "X_nh3 initial", 
                "X_o2 initial",
                "X_he initial",
                "(NH3/O2)",
                "T (C) final",
                "Rtol",
                "Atol",
                "reactor type",
            ]
            + gas.species_names
            + surf.species_names
            + gas_ROP_str
            + gas_surf_ROP_str
            + surf_ROP_str
            + gasrxn_ROP_str
            + surfrxn_ROP_str
            + sens_list
        )

    else:
        writer.writerow(
            [
                "T (C)",
                "P (atm)",
                "V (M^3/s)",
                "X_nh3 initial",
                "X_o2 initial",
                "X_he initial",
                "(NH3/O2)",
                "T (C) final",
                "Rtol",
                "Atol",
                "reactor type",
            ]
            + gas.species_names
            + surf.species_names
            + gas_ROP_str
            + gas_surf_ROP_str
            + surf_ROP_str
            + gasrxn_ROP_str
            + surfrxn_ROP_str
        )

    # writer_csp.writerow(
    #     ["iter", "t", "dt", "Density[kg/m3]", "Pressure[Pascal]", "Temperature[K]",]
    #     + gas.species_names
    #     + surf.species_names
    # )

    t = 0.0
    dt = 0.1
    iter_ct = 0
    # run the simulation
    first_run = True

    while t < reactime:
        # save flux diagrams at beginning of run
        if first_run == True:
            save_flux_diagrams(gas, suffix=flux_path, timepoint="beginning")
            save_flux_diagrams(surf, suffix=flux_path, timepoint="beginning")
            first_run = False
        t += dt
        sim.advance(t)
#         if t % 10 < 0.01:

        if sensitivity:
            # get sensitivity for sensitive species i (e.g. methanol) in reaction j
            for i in sens_species:
                g_nrxn = gas.n_reactions
                s_nrxn = surf.n_reactions
                # g_nspec = gas.n_species
                # s_nspec = surf.n_species

                gas_sensitivities = [sim.sensitivity(i, j) for j in range(g_nrxn)]
                surf_sensitivities = [
                    sim.sensitivity(i, j) for j in range(g_nrxn, g_nrxn + s_nrxn)
                ]
                # gas_therm_sensitivities = [sim.sensitivity(i,j)
                # for j in range(g_nrxn+s_nrxn,g_nrxn+s_nrxn+g_nspec)]
                # surf_therm_sensitivities = [sim.sensitivity(i,j)
                # for j in range(g_nrxn+s_nrxn+g_nspec,g_nrxn+s_nrxn+g_nspec+s_nspec)]

                sensitivities_all = (
                    gas_sensitivities
                    + surf_sensitivities
                    # + gas_therm_sensitivities
                )

            writer.writerow(
                [
                    temp,
                    pressure,
                    volume_flow,
                    X_nh3,
                    X_o2,
                    X_he,
                    o2_ratio,
                    gas.T,
                    sim.rtol,
                    sim.atol,
                    reactor_type_str,
                ]
                + list(gas.X)
                + list(surf.X)
                + list(gas.net_production_rates)
                + list(surf.net_production_rates)
                + list(gas.net_rates_of_progress)
                + list(surf.net_rates_of_progress)
                + sensitivities_all
            )

        else:
            writer.writerow(
                [
                    temp,
                    pressure,
                    volume_flow,
                    X_nh3,
                    X_o2,
                    X_he,
                    o2_ratio,
                    gas.T,
                    sim.rtol,
                    sim.atol,
                    reactor_type_str,
                ]
                + list(gas.X)
                + list(surf.X)
                + list(gas.net_production_rates)
                + list(surf.net_production_rates)
                + list(gas.net_rates_of_progress)
                + list(surf.net_rates_of_progress)
            )

        # writer_csp.writerow(
        #     [
        #         iter_ct,
        #         sim.time,
        #         dt,
        #         gas.density,
        #         gas.P,
        #         gas.T,
        #     ]
        #     + list(gas.X)
        #     + list(surf.X)
        # )

        iter_ct += 1

    outfile.close()
    # outfile_csp.close()

    # save flux diagrams at the end of the run
    save_flux_diagrams(gas, suffix=flux_path, timepoint="end")
    save_flux_diagrams(surf, suffix=flux_path, timepoint="end")
    return


def run_reactor_ss(
    cti_file,
    t_array=[548],
    p_array=[1],
    v_array=[2.7155e-8],
    o2_array=[0.88],
    nh3_array=[0.066],
    rtol=1.0e-11,
    atol=1.0e-22,
    reactor_type=0,
    energy="off",
    sensitivity=False,
    sensatol=1e-6,
    sensrtol=1e-6,
    reactime=1e5,
):
    ''' 
    run the reactor to steady state. saves a single CSV output file with one row of data. 
    results saved in new folder marked "steady state" under reactor type
    
    '''

    import pandas as pd
    import numpy as np
    import time
    import cantera as ct
    from matplotlib import pyplot as plt
    import csv
    import math
    import os
    import sys
    import re
    import itertools
    import logging
    from collections import defaultdict
    import git

    try:
        array_i = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    except TypeError:
        array_i = 0

    # get git commit hash and message

    repo = git.Repo("/work/westgroup/lee.ting/cantera/ammonia")
    git_sha = str(repo.head.commit)[0:6]
    git_msg = str(repo.head.commit.message)[0:20].replace(" ", "_").replace("'", "_")

    # this should probably be outside of function
    settings = list(itertools.product(t_array, p_array, v_array, o2_array, nh3_array))

    # constants
    pi = math.pi

    # set initial temps, pressures, concentrations
    temp = settings[array_i][0]  # kelvin
    temp_str = str(temp)[0:3]
    pressure = settings[array_i][1] * ct.one_atm  # Pascals

    X_o2 = settings[array_i][3]
    x_O2_str = str(X_o2)[0:3].replace(".", "_")
    if X_o2 == 0.88:
        X_he = 0.054
    else:
        X_he = 0
    X_nh3 = (1 - (X_o2 + X_he)) * (settings[array_i][4])
    x_NH3_str = str(X_nh3)[0:8].replace(".", "_")
    
    mw_nh3 = 17.0306e-3  # [kg/mol]
    mw_o2 = 31.999e-3  # [kg/mol]
    mw_he = 4.002602e-3  # [kg/mol]

    o2_ratio = X_nh3 / X_o2

    # CO/CO2/H2/H2: typical is
    concentrations_rmg = {"O2(2)": X_o2, "NH3(6)": X_nh3, "He": X_he}

    # initialize cantera gas and surface
    gas = ct.Solution(cti_file, "gas")

    # surf_grab = ct.Interface(cti_file,'surface1_grab', [gas_grab])
    surf = ct.Interface(cti_file, "surface1", [gas])

    # gas_grab.TPX =
    gas.TPX = temp, pressure, concentrations_rmg
    surf.TP = temp, pressure

    # create gas inlet
    inlet = ct.Reservoir(gas)

    # create gas outlet
    exhaust = ct.Reservoir(gas)

    # Reactor volume
    rradius = 1.4e-4 #140µm to 0.00014m
    rlength = 9e-3 #9mm to 0.009m
    rvol = (rradius ** 2) * pi * rlength / 2

    # Catalyst Surface Area
    site_density = (surf.site_density * 1000)  # [mol/m^2]cantera uses kmol/m^2, convert to mol/m^2
    cat_area = rradius * 2 / 2 * pi * rlength # [m^3]

    # reactor initialization
    if reactor_type == 0:
        r = ct.Reactor(gas, energy=energy)
        reactor_type_str = "Reactor"
    elif reactor_type == 1:
        r = ct.IdealGasReactor(gas, energy=energy)
        reactor_type_str = "IdealGasReactor"
    elif reactor_type == 2:
        r = ct.ConstPressureReactor(gas, energy=energy)
        reactor_type_str = "ConstPressureReactor"
    elif reactor_type == 3:
        r = ct.IdealGasConstPressureReactor(gas, energy=energy)
        reactor_type_str = "IdealGasConstPressureReactor"

    rsurf = ct.ReactorSurface(surf, r, A=cat_area)
    r.volume = rvol
    surf.coverages = "X(1):1.0"

    # flow controllers (Graaf measured flow at 293.15 and 1 atm)
    one_atm = ct.one_atm
    FC_temp = 293.15
    volume_flow = settings[array_i][2]  # [m^3/s]
    molar_flow = volume_flow * one_atm / (8.3145 * FC_temp)  # [mol/s]
    mass_flow = molar_flow * (X_nh3 * mw_nh3 + X_o2 * mw_o2 + X_he * mw_he)  # [kg/s]
    mfc = ct.MassFlowController(inlet, r, mdot=mass_flow)

    # A PressureController has a baseline mass flow rate matching the 'master'
    # MassFlowController, with an additional pressure-dependent term. By explicitly
    # including the upstream mass flow rate, the pressure is kept constant without
    # needing to use a large value for 'K', which can introduce undesired stiffness.
    outlet_mfc = ct.PressureController(r, exhaust, master=mfc, K=0.01)

    # initialize reactor network
    sim = ct.ReactorNet([r])

    # set relative and absolute tolerances on the simulation
    sim.rtol = 1.0e-11
    sim.atol = 1.0e-22

    #################################################
    # Run single reactor
    #################################################

    # round numbers so they're easier to read
    # temp_str = '%s' % '%.3g' % tempn

    cat_area_str = "%s" % "%.3g" % cat_area
    results_path = (
        os.path.dirname(os.path.abspath(__file__))
        + f"/{git_sha}_{git_msg}/{reactor_type_str}/steady_state/{temp_str}/results"
    )

    flux_path = (
        os.path.dirname(os.path.abspath(__file__))
        + f"/{git_sha}_{git_msg}/{reactor_type_str}/steady_state/{temp_str}/flux_diagrams/{x_O2_str}/{x_NH3_str}"
    )
    try:
        os.makedirs(results_path, exist_ok=True)
    except OSError as error:
        print(error)

    try:
        os.makedirs(flux_path, exist_ok=True)
    except OSError as error:
        print(error)

    gas_ROP_str = [i + " ROP [kmol/m^3 s]" for i in gas.species_names]

    # surface ROP reports gas and surface ROP. these values are not redundant.
    gas_surf_ROP_str = [i + " surface ROP [kmol/m^2 s]" for i in gas.species_names]
    surf_ROP_str = [i + " ROP [kmol/m^2 s]" for i in surf.species_names]

    gasrxn_ROP_str = [i + " ROP [kmol/m^3 s]" for i in gas.reaction_equations()]
    surfrxn_ROP_str = [i + " ROP [kmol/m^2 s]" for i in surf.reaction_equations()]

    output_filename = (
        results_path
        + f"/Spinning_basket_area_{cat_area_str}_energy_{energy}"
        + f"_temp_{temp}_O2_{x_O2_str}_NH3_{x_NH3_str}.csv"
    )
    
    outfile = open(output_filename, "w")
    writer = csv.writer(outfile)
    
    writer.writerow(
        [
            "T (C)",
            "P (atm)",
            "V (M^3/s)",
            "X_nh3 initial",
            "X_o2 initial",
            "X_he initial",
            "(NH3/O2)",
            "T (C) final",
            "Rtol",
            "Atol",
            "reactor type",
        ]
        + gas.species_names
        + surf.species_names
        + gas_ROP_str
        + gas_surf_ROP_str
        + surf_ROP_str
        + gasrxn_ROP_str
        + surfrxn_ROP_str
    )
        
    # Run Simulation to steady state
    sim.advance_to_steady_state()
    
    # Record steady state data to CSV
    writer.writerow(
        [
            temp,
            pressure,
            volume_flow,
            X_nh3,
            X_o2,
            X_he,
            o2_ratio,
            gas.T,
            sim.rtol,
            sim.atol,
            reactor_type_str,
        ]
        + list(gas.X)
        + list(surf.X)
        + list(gas.net_production_rates)
        + list(surf.net_production_rates)
        + list(gas.net_rates_of_progress)
        + list(surf.net_rates_of_progress)
    )

  

    outfile.close()

    # save flux diagrams at the end of the run
    save_flux_diagrams(gas, suffix=flux_path, timepoint="end")
    save_flux_diagrams(surf, suffix=flux_path, timepoint="end")
    return

#######################################################################
# Input Parameters for combustor
#######################################################################

# filepath for writing files
# cti_file = os.path.dirname(os.path.abspath(__file__)) +'/chem_annotated.cti'
cti_file = "/work/westgroup/lee.ting/cantera/ammonia/base/cantera/chem_annotated.cti"

# Reactor settings arrays for run
Temps = [400, 550, 700]
Pressures = [1, 1, 1, 1]
volume_flows = [5.8333e-5, 5.8333e-5, 5.8333e-5] # [m^3/s] 
#3500 Ncm3/min = 3500/e6/60 m3/s = 5.8333e-5

# NH3/O2 = 0.068
### why h2 has 4 values but co2/co has 9 values
O2_fraction = [0.1, 0.3, 0.6, 0.9] #O2 partial pressure, 0.10–0.88 atm
NH3_fraction = [0.01, 0.0275, 0.045, 0.0625, 0.08, 0.0975, 0.115, 0.1325, 0.15] #NH3 partial pressure, 0.01–0.12 atm

# reaction time
reactime = 1e3

# sensitivity settings
sensitivity = False
sensatol = 1e-6
sensrtol = 1e-6

run_reactor(
    cti_file=cti_file,
    t_array=Temps,
    reactor_type=1,
    o2_array=O2_fraction,
    nh3_array=NH3_fraction,
    sensitivity=sensitivity,
    sensatol=sensatol,
    sensrtol=sensrtol,
    reactime=reactime,
)

# could put a try catch here, but 
# this way it will run the regular sim
# before attempting the steady state solver
run_reactor_ss(
    cti_file=cti_file,
    t_array=Temps,
    reactor_type=1,
    o2_array=O2_fraction,
    nh3_array=NH3_fraction,
)