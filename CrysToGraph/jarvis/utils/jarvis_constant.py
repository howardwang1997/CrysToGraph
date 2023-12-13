DATASETS_LEN = {
    'jarvis_supercon': 1058,
    'jarvis_exfoliation': 4527,
    'jarvis_magnetization': 6351,
    'jarvis_2d_gap': 3520,
    'jarvis_2d_e_tot': 3520,
    'jarvis_2d_e_fermi': 3520,
    'jarvis_qmof_energy': 20425,
    'jarvis_co2_adsp': 137652,
    'jarvis_surface': 137652,
    'jarvis_vacancy': 530,
}

DATASETS_RESULTS = {
    'jarvis_supercon': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_exfoliation': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_magnetization': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_2d_gap': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_2d_e_tot': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_2d_e_fermi': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_qmof_energy': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_co2_adsp': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_surface': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
    'jarvis_vacancy': {'fold_0': [], 'fold_1': [], 'fold_2': [], 'fold_3': [], 'fold_4': []},
}

DATASETS_MAP = {
    'jarvis_supercon': {'path': 'jarvis_datasets/jarvis_epc_data_figshare_1058.json', 'label': 'Tc'},
    'jarvis_exfoliation': {'path': 'jarvis_datasets/2dmatpedia_exfoliation.json', 'label': 'exfoliation_energy_per_atom'},
    'jarvis_magnetization':  {'path': 'jarvis_datasets/twodmatpedia.json', 'label': 'total_magnetization'},
    'jarvis_2d_gap':  {'path': 'jarvis_datasets/c2db_atoms.json', 'label': 'gap'},
    'jarvis_2d_e_tot':  {'path': 'jarvis_datasets/c2db_atoms.json', 'label': 'etot'},
    'jarvis_2d_e_fermi':  {'path': 'jarvis_datasets/c2db_atoms.json', 'label': 'efermi'},
    'jarvis_qmof_energy':  {'path': 'jarvis_datasets/qmof_db.json', 'label': 'energy_total'},
    'jarvis_co2_adsp':  {'path': 'jarvis_datasets/hmof_db_9_18_2021.json', 'label': 'max_co2_adsp'},
    'jarvis_surface':  {'path': 'jarvis_datasets/hmof_db_9_18_2021.json', 'label': 'surface_area_m2g'},
    'jarvis_vacancy':  {'path': 'jarvis_datasets/vacancydb.json', 'label': 'ef'},
}
