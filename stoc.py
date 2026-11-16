import numpy as np

def ordre_random(jour):
    mardi = [
        "Arthur/Ethan/moi même",
        "Jimmy/Kylian",
        "Matéo/Esteban",
        "Florian/Maëlys",
        "Eliot",
        "Dylan/Maxime",
        "Tristan/Eliott/Clarck",
    ]

    jeudi = [
        "Yako/Thomas/Felix (lock ce jour là)",
        "Benjamin/Nicolas",
        "Kenan/Noah",
        "Khoren/Quentin",
        "Aboubakar/Hamadi/Valentin",
    ]

    groupes = mardi if jour == "mardi" else jeudi
    ordre = np.random.permutation(groupes)

    print(f"\nOrdre de passage {jour.capitalize()} :\n")
    for i, groupe in enumerate(ordre, 1):
        print(f"{i}. {groupe}")
    print()

ordre_random("mardi")
ordre_random("jeudi")