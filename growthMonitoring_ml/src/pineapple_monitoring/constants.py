HEALTH_CLASSES = ["healthy", "nitrogen_deficiency", "water_stress"]
MONTH_CLASSES = [f"M{i}" for i in range(1, 13)]

HEALTH_TO_IDX = {name: i for i, name in enumerate(HEALTH_CLASSES)}
IDX_TO_HEALTH = {i: name for name, i in HEALTH_TO_IDX.items()}

MONTH_TO_IDX = {name: i for i, name in enumerate(MONTH_CLASSES)}
IDX_TO_MONTH = {i: name for name, i in MONTH_TO_IDX.items()}
