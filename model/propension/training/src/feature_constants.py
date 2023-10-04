# Numeric columns to impute with 0
NUM_COLS = [
    "ind_cliente",
    "ctd_productos",
    "flg_eps",
    "flg_rrgg",
    "flg_vida",
    "flg_vehi",
    "mto_prima_contable_usd",
    "flg_fue_cliente_rimac",
    "val_scoring_ingreso",
    "mto_max_linea_tc",
    "mto_saldo_tc_sbs",
    "mto_saldo_sbs",
    "flg_tiene_vehiculo",
    "flg_escliente",
]

# Numeric columns to impute with median
AGE_COL = ["num_edad"]

# One Hot encoding columns
OHE_COLS = ["des_lima_prov"]

# Ordinal columns
NSE_COL = ["nse"]
RCC_COL = ["cal_gral"]
DES_CONO_COL = ["des_cono_agrup_nuevo"]
COMBO_COL = ["des_combo_productos"]

# Ordinal columns category order
NSE_CATEGORY_ORDER = ["OTRO", "E", "D", "C2", "C1", "B2", "B1", "A2", "A1"]
RCC_CATEGORY_ORDER = [
    "OTRO",
    "PERDIDA",
    "DUDOSO",
    "DEFICIENTE",
    "CON PROBLEMAS POTENCIALES",
    "NORMAL",
]
LIMA_PROV_CATEGORY_ORDER = ["OTRO", "LIMA-CALLAO", "PROVINCIA"]
