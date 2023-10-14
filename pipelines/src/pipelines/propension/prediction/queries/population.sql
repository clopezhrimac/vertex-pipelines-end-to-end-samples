DROP TABLE IF EXISTS `{{ population_dataset }}.{{ population_table }}`;
CREATE TABLE `{{ population_dataset }}.{{ population_table }}` AS (
    SELECT DATE('{{ prediction_period }}') AS periodo, id_persona
    FROM `{{ feature_store_dataset }}.{{ source_table }}`
    TABLESAMPLE SYSTEM (0.01 PERCENT)
    WHERE tip_persona = 'PN'
);
