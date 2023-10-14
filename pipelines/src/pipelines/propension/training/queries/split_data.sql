DROP TABLE IF EXISTS `{{ split_data_dataset }}.{{ train_table }}`;
CREATE TABLE `{{ split_data_dataset }}.{{ train_table }}` AS (
    SELECT *
    FROM
        `{{ enriched_dataset }}.{{ enriched_table }}`
    WHERE
        DATE({{ filter_column }}) BETWEEN
        DATE('{{ train_start_date }}') AND
        DATE('{{ train_final_date }}')
);

DROP TABLE IF EXISTS `{{ split_data_dataset }}.{{ validation_table }}`;
CREATE TABLE `{{ split_data_dataset }}.{{ validation_table }}` AS (
    SELECT *
    FROM
        `{{ enriched_dataset }}.{{ enriched_table }}`
    WHERE
        DATE({{ filter_column }}) BETWEEN
        DATE('{{ validation_start_date }}') AND
        DATE('{{ validation_final_date }}')
);

DROP TABLE IF EXISTS `{{ split_data_dataset }}.{{ test_table }}`;
CREATE TABLE `{{ split_data_dataset }}.{{ test_table }}` AS (
    SELECT *
    FROM
        `{{ enriched_dataset }}.{{ enriched_table }}`
    WHERE
        DATE({{ filter_column }}) BETWEEN
        DATE('{{ test_start_date }}') AND
        DATE('{{ test_final_date }}')
);
