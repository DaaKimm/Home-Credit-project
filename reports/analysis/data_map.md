# Home Credit raw-data map

Source dictionary: `data/raw/HomeCredit_columns_description.csv`.  A `docs/`
directory is not present in this checkout. Counts below exclude CSV headers.
This is a schema/identifier audit only; it does not profile features or perform
EDA. “Applicants covered” is the number and share of the 307,511
`application_train` `SK_ID_CURR` values with one or more rows in the table.

`application_train` and `application_test` are disjoint partitions, so a
relationship to “applications” below means the union of those two tables.

| Table | Grain and row count | Primary key (unique?) | Foreign keys and join cardinality | Train applicants covered |
|---|---|---|---|---|
| `application_train.csv` | One current loan application with its outcome; **307,511** rows. | `SK_ID_CURR`; **yes** (307,511 distinct). | None. Parent application entity. | **307,511 (100.00%)** |
| `application_test.csv` | One current, unlabeled loan application; **48,744** rows. | `SK_ID_CURR`; **yes** (48,744 distinct). | None. Disjoint application partition. | **0 (0.00%)** — intentionally disjoint from train. |
| `bureau.csv` | One external Credit Bureau credit record associated with a current application; **1,716,428** rows. | `SK_ID_BUREAU`; **yes**. | `SK_ID_CURR` → applications: one application to **0..many** bureau records. Every value resolves to the application union. | **263,491 (85.69%)** |
| `bureau_balance.csv` | One monthly status observation for a bureau credit; **27,299,925** rows. | (`SK_ID_BUREAU`, `MONTHS_BALANCE`); **yes**. | Candidate `SK_ID_BUREAU` → `bureau.SK_ID_BUREAU`: one bureau record to **0..many** monthly statuses. It is not fully enforced: **3,120,184 rows** do not resolve to the supplied `bureau` table. | **92,231 (29.99%) confirmed** through resolving bureau records; orphaned bureau IDs cannot be attributed to an applicant. |
| `previous_application.csv` | One previous Home Credit loan application; **1,670,214** rows. | `SK_ID_PREV`; **yes**. | `SK_ID_CURR` → applications: one application to **0..many** previous applications. Every value resolves to the application union. | **291,057 (94.65%)** |
| `POS_CASH_balance.csv` | One monthly POS/cash-loan status observation for a previous credit; **10,001,358** rows. | (`SK_ID_PREV`, `MONTHS_BALANCE`); **yes**. | `SK_ID_CURR` → applications: **0..many** rows per application (all resolve). Candidate `SK_ID_PREV` → `previous_application`: one previous application to **0..many** observations, but **340,561 rows** have no matching supplied previous-application ID. | **289,444 (94.12%)** |
| `credit_card_balance.csv` | One monthly credit-card balance snapshot for a previous credit; **3,840,312** rows. | (`SK_ID_PREV`, `MONTHS_BALANCE`); **yes**. | `SK_ID_CURR` → applications: **0..many** rows per application (all resolve). Candidate `SK_ID_PREV` → `previous_application`: one previous application to **0..many** snapshots, but **1,082,816 rows** do not resolve. | **86,905 (28.26%)** |
| `installments_payments.csv` | One recorded scheduled/actual installment-payment observation for a previous credit; **13,605,401** rows. | No unique primary key is supplied. The natural calendar tuple (`SK_ID_PREV`, `NUM_INSTALMENT_VERSION`, `NUM_INSTALMENT_NUMBER`) is **not unique** (12,951,918 distinct tuples), so it must not be used as a PK. | `SK_ID_CURR` → applications: **0..many** rows per application (all resolve). Candidate `SK_ID_PREV` → `previous_application`: one previous application to **0..many** payment observations, but **1,250,826 rows** do not resolve. | **291,643 (94.84%)** |
| `sample_submission.csv` | One scoring-template row for a test application; **48,744** rows. | `SK_ID_CURR`; **yes**. | `SK_ID_CURR` → `application_test.SK_ID_CURR`: **1:1**; all and only test IDs are present. | **0 (0.00%)** — it is a test-only template. |

## Dictionary contradictions

The dictionary states that `DAYS_EMPLOYED` is “how many days **before** the
application” the client started the current job. That requires a non-positive
relative-day value. Both application tables instead contain the positive
sentinel value **365243**:

| Column | Contradictory observed value | Affected rows |
|---|---:|---:|
| `application_train.DAYS_EMPLOYED` | `365243` days (a future, not before-application, value) | 55,374 |
| `application_test.DAYS_EMPLOYED` | `365243` days (a future, not before-application, value) | 9,274 |

No other contradiction was found among the dictionary's explicit value rules
(binary encodings and the documented 1/2/3 region-rating domains). The
unresolved identifiers noted above are referential-integrity exceptions, not
value-domain contradictions in the column dictionary.
