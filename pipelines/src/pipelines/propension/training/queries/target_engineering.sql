DROP TABLE IF EXISTS `{{ target_engineering_dataset }}.{{ target_engineering_table }}`;
CREATE TABLE `{{ target_engineering_dataset }}.{{ target_engineering_table }}` AS (
    WITH
    filtered_leads AS (
        SELECT
            periodo,
            id_lead,
            id_persona,
            tip_documento,
            num_documento,
            gst_ind_cuenta_ganado,
            ind_cliente
        FROM
            `{{ source_dataset }}.{{ source_table }}`
        WHERE
            des_producto_filtro IN (
                'Salud (AMI)',
                'Salud (Oro)',
                'Salud (Red Hospitalaria)',
                'Salud (Flexi)'
            )
            AND ind_cuenta_gestionable = 1
            AND id_persona IS NOT NULL
            AND DATE({{ filter_column }}) BETWEEN
            DATE_SUB(DATE('{{ filter_start_value }}'), INTERVAL 12 MONTH) AND
            DATE_SUB(DATE('{{ filter_start_value }}'), INTERVAL 1 MONTH)
    )

    SELECT
        periodo,
        id_persona,
        ind_cliente,
        MAX(gst_ind_cuenta_ganado) AS `{{ target_column }}`
    FROM
        filtered_leads
    GROUP BY
        periodo,
        id_persona,
        ind_cliente
);
