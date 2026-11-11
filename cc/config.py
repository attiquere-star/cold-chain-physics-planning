"""Model and world parameters. All values are illustrative and literature-informed;
see the parameter table in the paper for the sources and the items marked [VERIFY]."""
import copy
import numpy as np

BASE = dict(
    DT=0.25,            # planning/simulation step (days)
    SPD=4,              # steps per day
    H_DAYS=7,           # planning horizon (days)
    P=2,                # products: 0 = fresh produce, 1 = chilled dairy
    SL_DAYS=[7, 12],    # shelf life from production (days)
    K_REF=[0.045, 0.020],  # first-order quality-loss rate at 4 degC (1/day)
    Q10=[2.5, 2.0],     # temperature sensitivity of loss rate
    T_SET=4.0,          # refrigeration set point (degC)
    # suppliers (raw material, one unit per product unit), lead time one step
    S=3, C_SUP=[3.0, 3.3, 3.7], CAP_SUP=[300.0, 300.0, 450.0], F_SUP=[180.0, 120.0, 60.0],
    PROD_CAP=650.0, C_PROD=1.0,
    # long-haul route plant -> DC
    L0=900.0, V0=480.0, CAP0=1000.0, C_TRUCK0=1100.0, EF0=0.95, NMAX0=2,
    # regional routes DC -> customers
    C=4, LC=[150.0, 250.0, 350.0, 450.0], VC=380.0, CAPC=350.0,
    C_TRUCKC_FIX=120.0, C_TRUCKC_KM=1.2, EFC=0.75, NMAXC=2,
    MU=[[120, 80], [90, 70], [150, 60], [70, 90]],  # base daily demand
    # holding (per unit-day)
    H_RAW=0.01, H_DC=0.06, H_TR=0.03, H_CUS=0.10,
    # energy (kWh per unit-day at 20 degC ambient), price and grid factor
    E_DC=0.05, E_TR=0.08, P_KWH=0.12, GRID_EF=0.58,
    CARBON=0.10,        # carbon price per kg CO2
    DISPOSAL=0.4,       # disposal cost per spoiled/expired unit
    PI_B=0.6,           # backlog penalty per unit-day
    RHO=0.5,            # backlog abandonment rate (1/day)
    PI_LOST=12.0,       # lost-sales penalty per unit
    KAPPA=0.08,         # retail stock loss rate (1/day)
    V_TERM=3.6,         # terminal value per unit held at horizon end
    MIP_GAP=0.005, TIME_LIMIT=8.0, INT_DAYS=1,
)

def cfg(**over):
    c = copy.deepcopy(BASE)
    c.update(over)
    return c

def loss_rate(c, p, temp):
    """Q10 (Arrhenius-type) first-order quality-loss rate, 1/day."""
    return c['K_REF'][p] * c['Q10'][p] ** ((np.asarray(temp) - c['T_SET']) / 10.0)
