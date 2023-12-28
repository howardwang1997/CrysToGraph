from .tasks import Task


DATASETS_TASKS = {
    'jarvis_supercon': Task('jarvis_supercon'),
    'jarvis_exfoliation': Task('jarvis_exfoliation'),
    'jarvis_magnetization': Task('jarvis_magnetization'),
    'jarvis_2d_gap': Task('jarvis_2d_gap'),
    'jarvis_2d_e_tot': Task('jarvis_2d_e_tot'),
    'jarvis_2d_e_fermi': Task('jarvis_2d_e_fermi'),
    'jarvis_qmof_energy': Task('jarvis_qmof_energy'),
    'jarvis_co2_adsp': Task('jarvis_co2_adsp'),
    'jarvis_surface': Task('jarvis_surface'),
    'jarvis_vacancy': Task('jarvis_vacancy'),
}
