DROP TABLE IF EXISTS `{{ enriched_dataset }}.{{ enriched_table }}`;
CREATE TABLE `{{ enriched_dataset }}.{{ enriched_table }}` AS (
    SELECT
        t.periodo,
        t.id_persona,
        CASE WHEN pc.id_persona IS NULL THEN 0 ELSE 1 END AS ind_cliente,
        pc.ctd_productos,
        pc.flg_eps,
        pc.flg_rrgg,
        pc.flg_vida,
        pc.flg_vehi,
        pc.mto_prima_contable_usd,
        pc.des_combo_productos,
        pc.flg_fue_cliente_rimac,
        pd.val_scoring_ingreso,
        pd.nse,
        pd.num_edad,
        pic.mto_max_linea_tc,
        pic.cal_gral,
        pic.mto_saldo_tc_sbs,
        pic.mto_saldo_sbs,
        pp.flg_tiene_vehiculo,
        pp.flg_escliente,
        ps.des_lima_prov,
        ps.des_cono_agrup_nuevo
    FROM `{{ population_dataset }}.{{ population_table }}` AS t
    LEFT JOIN `{{ feature_store_dataset }}.persona__cliente` AS pc
        ON DATE_SUB(t.periodo, INTERVAL 1 MONTH) = pc.periodo AND t.id_persona = pc.id_persona
    LEFT JOIN `{{ feature_store_dataset }}.persona__informacion_crediticia` AS pic
        ON DATE_SUB(t.periodo, INTERVAL 1 MONTH) = pic.periodo AND t.id_persona = pic.id_persona
    LEFT JOIN `{{ feature_store_dataset }}.persona__prospeccion` AS pp
        ON DATE_SUB(t.periodo, INTERVAL 1 MONTH) = pp.periodo AND t.id_persona = pp.id_persona
    LEFT JOIN `{{ feature_store_dataset }}.persona__static` AS ps
        ON t.id_persona = ps.id_persona
    LEFT JOIN `{{ feature_store_dataset }}.persona__dynamic_pn` AS pd
        ON DATE_SUB(t.periodo, INTERVAL 1 MONTH) = pd.periodo AND t.id_persona = pd.id_persona
);
