from z3 import *

# Variables
debtor_age_at_debt = Int('debtor_age_at_debt')
debt_unrecoverable = Bool('debt_unrecoverable')
debtor_entitled    = Bool('debtor_entitled')
debtor_alive       = Bool('debtor_alive')
boan_on_mbr        = Bool('boan_on_mbr')
debt_amount        = Real('debt_amount')
debt_delinquent    = Bool('debt_delinquent')
debtor_type        = String('debtor_type')
payment_type       = String('payment_type')

# Criteria
criterion_a = debtor_age_at_debt >= 18
criterion_b = debt_unrecoverable == True
criterion_c = debtor_entitled == False
criterion_d = debtor_alive == True
criterion_e = boan_on_mbr == True
criterion_f = debt_amount >= 25.00
criterion_g = debt_delinquent == True
criterion_h = Not(And(debtor_type == StringVal("BIC_C"),
                      payment_type != StringVal("TRO")))

# Composite node
offset_eligible = And(
    criterion_a, criterion_b, criterion_c, criterion_d,
    criterion_e, criterion_f, criterion_g, criterion_h
)