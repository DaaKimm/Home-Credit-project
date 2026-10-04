# Repayment-risk features for thin-file applicants: evidence and data feasibility

## Scope and reading rule

This is a **feature-availability assessment**, not an EDA or a model.  No
values, rates, aggregates, or scores were computed.  “Can build” means that
the raw fields necessary to derive the stated feature exist, subject to the
join and coverage limitations documented in `data_map.md`; it does not mean
that the feature is appropriate, predictive, lawful in every jurisdiction, or
available for every applicant.

“Thin file” here means little or no usable traditional credit history. That is
an important distinction: a feature made from `bureau.csv` or prior Home Credit
accounts is useful where present, but is intrinsically absent for some of the
population the question is about. The CFPB's updated estimate distinguishes
credit-invisible people from people with records that are unscored because the
file is insufficient or stale [CFPB, 2025](https://www.consumerfinance.gov/data-research/research-reports/technical-correction-and-update-to-the-cfpbs-credit-invisibles-estimate/).

## What lenders use (and why)

Traditional underwriting combines willingness/ability signals from credit
reports with capacity information. For example, the current credit-card
ability-to-pay rule requires consideration of income or assets and current
obligations, and permits consumer reports, credit scores, and other factors
[12 CFR 1026.51](https://www.consumerfinance.gov/rules-policy/regulations/1026/51/).
The conventional score dimensions are payment history, debt/use of available
credit, account age, recent applications, and credit mix
[Experian lender/CRA disclosure](https://www.experian.com/blogs/ask-experian/how-is-your-credit-score-determined/).

For thin files, lenders and data providers may add cash-flow, recurring bill,
employment/education, telecom, or digital-footprint information. The CFPB
lists rent, mobile and cable bills, and bank deposits/withdrawals/transfers as
examples; it also identifies education, occupation, social-media, and website
interaction data as types considered by some lenders
[CFPB alternative-data explainer](https://www.consumerfinance.gov/archive/blog/using-alternative-data-evaluate-creditworthiness/).
The joint banking-agency statement specifically recognizes cash-flow data from
reliable bank records as a form of alternative underwriting data
[interagency statement](https://ncua.gov/regulation-supervision/letters-credit-unions-other-guidance/interagency-statement-use-alternative-data-credit-underwriting).
The empirical literature finds signal in digital footprints
[Berg et al., *Review of Financial Studies*, 2020](https://doi.org/10.1093/rfs/hhz099)
and in mobile-phone behavior for people with no credit history
[Björkegren & Grissen, *World Bank Economic Review*, 2020](https://academic.oup.com/wber/article/34/3/618/5622690).

These are evidence of what may be used, not a recommendation to use every
item. The CFPB cautions that alternative data can create unlawful-discrimination
risk through proxy relationships. Protected characteristics and close proxies
need a separate legal, fairness, explainability, and governance review
[CFPB](https://www.consumerfinance.gov/archive/blog/using-alternative-data-evaluate-creditworthiness/).

## Mapping to this project’s data

All application-level fields below occur in both `application_train.csv` and
`application_test.csv`, unless stated otherwise. Use `SK_ID_CURR` to attach
the many-row tables to an application. For bureau monthly history use
`bureau_balance.SK_ID_BUREAU` → `bureau.SK_ID_BUREAU` → `SK_ID_CURR`; for
Home Credit prior-account tables use `SK_ID_CURR` (or `SK_ID_PREV` where a
prior-account history must be attached). `data_map.md` reports missing related
records and orphaned child keys, so joins must preserve the application and
explicitly represent missing history rather than silently treating it as good
history.

| Risk feature / feature family | Evidence-based rationale | Can this data build it? Exact tables and columns | Thin-file limitation / caveat |
|---|---|---|---|
| **Prior payment performance**: late-payment incidence, severity, recency, on-time share, paid-vs-scheduled amount | Payment history is a central conventional credit-score factor. | **Yes, where prior history exists.** `bureau.csv`: `CREDIT_DAY_OVERDUE`, `AMT_CREDIT_MAX_OVERDUE`, `AMT_CREDIT_SUM_OVERDUE`, `DAYS_CREDIT_UPDATE`; `bureau_balance.csv`: `MONTHS_BALANCE`, `STATUS`. Internal history: `installments_payments.csv`: `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`, `AMT_INSTALMENT`, `AMT_PAYMENT`; `POS_CASH_balance.csv`: `MONTHS_BALANCE`, `SK_DPD`, `SK_DPD_DEF`, `NAME_CONTRACT_STATUS`; `credit_card_balance.csv`: `MONTHS_BALANCE`, `SK_DPD`, `SK_DPD_DEF`, `NAME_CONTRACT_STATUS`. | It cannot supply a traditional history for an applicant with no bureau/prior-account rows. `bureau_balance` has the unresolved bureau IDs noted in `data_map.md`; do not assign them to an applicant. |
| **Current debt burden / outstanding balance** | Debt and current obligations are conventional risk and ability-to-pay inputs; DTI is monthly debt payments divided by gross monthly income [CFPB](https://www.consumerfinance.gov/ask-cfpb/what-is-a-debt-to-income-ratio-en-1791/). | **Partly.** Applicant income: `application_{train|test}.csv.AMT_INCOME_TOTAL`; requested-loan payment: `AMT_ANNUITY`; external outstanding debt: `bureau.csv.AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM`, `AMT_ANNUITY`, `CREDIT_ACTIVE`; card balances: `credit_card_balance.csv.AMT_BALANCE`, `AMT_TOTAL_RECEIVABLE`, `AMT_INST_MIN_REGULARITY`. | A complete DTI **cannot** be built: the data lack all verified monthly obligations (e.g., rent, mortgages, child support) and income periodicity/verification. Ratios built here are partial proxies, not regulatory DTI. |
| **Revolving-credit utilization and available credit** | The balance-to-limit relationship is a conventional “amounts owed” signal. | **Yes, for observed cards/bureau limits.** `credit_card_balance.csv.AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL`, `AMT_TOTAL_RECEIVABLE`, `MONTHS_BALANCE`; `bureau.csv.AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM_LIMIT`, `CREDIT_ACTIVE`, `CREDIT_TYPE`. | Only cards/limits represented in supplied prior or bureau data; not a full cross-lender utilization measure for thin files. |
| **Credit depth, age, and mix**: number of accounts, oldest/recent account, active/closed share, revolving vs installment/product mix | Account age and mix are conventional score dimensions. | **Yes, when records exist.** `bureau.csv`: `DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `CREDIT_ACTIVE`, `CREDIT_TYPE`; `previous_application.csv`: `DAYS_DECISION`, `NAME_CONTRACT_TYPE`, `NAME_PORTFOLIO`, `NAME_CONTRACT_STATUS`, `CNT_PAYMENT`; `POS_CASH_balance.csv.CNT_INSTALMENT` and `CNT_INSTALMENT_FUTURE`; `credit_card_balance.csv.CNT_INSTALMENT_MATURE_CUM`. | Absent records are not proof of no credit; `bureau` covers only 85.69% of train applicants per `data_map.md`. |
| **Recent credit seeking / application velocity** | New credit and hard inquiries are conventional score dimensions; many recent applications can signal stress. | **Yes, with two distinct sources.** Current bureau enquiries: `application_train.csv` / `application_test.csv`: `AMT_REQ_CREDIT_BUREAU_HOUR`, `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`, `AMT_REQ_CREDIT_BUREAU_MON`, `AMT_REQ_CREDIT_BUREAU_QRT`, `AMT_REQ_CREDIT_BUREAU_YEAR`. Prior in-house applications: `previous_application.csv.DAYS_DECISION`, `NAME_CONTRACT_STATUS`, `NFLAG_LAST_APPL_IN_DAY`, `FLAG_LAST_APPL_PER_CONTRACT`, `SK_ID_PREV`. | The in-house stream is not market-wide enquiries; bureau-enquiry availability/missingness needs treatment. |
| **Requested credit capacity / payment-to-income proxy** | Underwriting evaluates the payment required under the proposed terms relative to income/assets and obligations. | **Partly.** Current application: `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_CONTRACT_TYPE`, with `AMT_INCOME_TOTAL`. Prior offer/terms: `previous_application.csv.AMT_APPLICATION`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_DOWN_PAYMENT`, `RATE_DOWN_PAYMENT`, `CNT_PAYMENT`, `NAME_YIELD_GROUP`. | Income is declared in the dictionary, not demonstrated as verified; no taxes, payroll, assets, or full expenses. This supports affordability proxies only. |
| **Employment and income stability** | Income and employment are named ability-to-repay inputs; occupation is an alternative-data type some lenders have considered. | **Partly.** `application_{train|test}.csv`: `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`. | No payroll deposits, employer verification, hours, job changes, or prior income. `DAYS_EMPLOYED=365243` is a documented sentinel contradiction in `data_map.md` and must be handled as missing/not-employed per a documented decision—not as tenure. |
| **Housing/residential stability and assets** | Lenders may consider assets; application information can provide stability/capacity context. | **Only proxies.** `application_{train|test}.csv`: `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `FLAG_OWN_CAR`, `OWN_CAR_AGE`, `DAYS_REGISTRATION`, `REG_REGION_NOT_LIVE_REGION`, `REG_CITY_NOT_LIVE_CITY`, `LIVE_CITY_NOT_WORK_CITY`, plus building fields such as `APARTMENTS_AVG`, `TOTALAREA_MODE`, `HOUSETYPE_MODE`. | No home value/equity, lease, rent-payment history, savings, or verified assets. Residential/geographic proxies pose meaningful fairness risk and should not be assumed to measure repayment ability. |
| **Household obligations / dependent load** | Household composition can affect residual income, but must be governed carefully. | **Partly and cautiously.** `application_{train|test}.csv.CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `AMT_INCOME_TOTAL`. | No spouse/household income, actual expenses, support payments, or joint liability. Family status may be legally sensitive; do not use it merely because it is present. |
| **Education / occupation attributes** | CFPB identifies education and occupation as alternative information some lenders have considered. | **Yes, as raw categorical attributes only.** `application_{train|test}.csv.NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`. | No credentials verification, work history, skills, or earnings trajectory. These variables can proxy protected traits; feasibility is not endorsement. |
| **Cash-flow resilience**: deposit regularity, balances, discretionary outflows, NSF/overdraft behavior | Cash-flow underwriting uses account inflows, outflows, and accumulated balances; CFPB reports it can add information beyond credit history. | **No.** There is no checking/savings account, transaction, payroll-deposit, merchant-spend, overdraft, or asset-balance table. `credit_card_balance` is a credit-account snapshot, not deposit-account cash flow. | This is a major missing thin-file feature family. Do not relabel card drawings/payments as bank cash flow. |
| **Rent, utility, cable, and mobile-bill payment history** | CFPB identifies rent, mobile-phone, and cable bills as alternative credit data, especially relevant to credit invisibles. | **No.** No rental ledger, utility/cable/telecom account, invoice, due date, payment, or service-status columns exist. | `FLAG_PHONE`, `FLAG_MOBIL`, `FLAG_EMAIL`, and `DAYS_LAST_PHONE_CHANGE` only describe contact/device availability or change; they are not telecom bill-payment history. |
| **Mobile-phone behavioral signal**: call/text/top-up/usage regularity | Peer-reviewed evidence shows behavioral features from mobile-phone usage can predict repayment among people without credit history. | **No.** No call-detail records, SMS, handset, SIM, top-up, location trail, or telecom usage data exist. | Do not infer this from the phone-contact flags or `DAYS_LAST_PHONE_CHANGE`; that would be a different and much weaker construct. |
| **Digital-footprint / online-application behavior** | Berg et al. find predictive information in website-access/registration digital footprints. | **No.** There are no device, IP, browser, clickstream, session, e-commerce, social-network, or online-form interaction fields. `WEEKDAY_APPR_PROCESS_START` and `HOUR_APPR_PROCESS_START` are application timestamps, not a digital footprint. | This absence is useful from a privacy/governance perspective, but it means the published feature family cannot be reproduced. |
| **Third-party external risk score** | Credit scores/consumer reports are expressly permitted additional factors in card ability-to-pay underwriting. | **Ambiguous / not reproducible as a known score.** `application_{train|test}.csv.EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are described only as “normalized score from external data source.” | The source, scale, timing, inputs, and permissible use are undocumented. They can be model inputs only after provenance, consent, adverse-action/explainability, and fairness review; the data cannot recreate or validate the external scores. |

## Practical conclusion

The data are rich for **traditional repayment behavior, debt, utilization,
credit age/mix, inquiry velocity, requested terms, and coarse application
stability proxies** when a prior record exists. They are not a full thin-file
alternative-data dataset: the three especially relevant gaps are **bank-account
cash flow**, **on-time noncredit bills (rent/utilities/telecom)**, and **actual
telecom/digital behavior**. For applicants with no bureau and no Home Credit
history, the buildable evidence contracts mainly to self-reported application
data and opaque external scores.

The supplied `TARGET` is deliberately excluded from every feature definition:
it is an outcome in `application_train.csv`, and `application_test.csv` has no
such column. It may be used later to train/evaluate a properly time-safe model,
but never as an applicant-time feature.
