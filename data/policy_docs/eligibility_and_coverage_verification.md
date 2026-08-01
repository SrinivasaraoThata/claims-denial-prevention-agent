# Member Eligibility and Coverage Verification

## Eligibility statuses
Each member record carries one of three eligibility statuses:
- `active` - coverage is in effect; claims for dates of service within the
  coverage window are eligible for payment.
- `lapsed` - coverage has ended due to non-payment of premium or missed
  renewal, but the member may be reinstated within a grace period defined by
  the plan.
- `terminated` - coverage has ended permanently (e.g. loss of eligibility,
  disenrollment). No claims are payable for dates of service after the
  coverage end date.

## Verification requirement
Eligibility must be verified as of the date of service, not the date the
claim is submitted. A claim for a service rendered while a member's coverage
was active is payable even if the member's status has since changed to
lapsed or terminated.

## Common denial reasons
- Date of service falls before the member's coverage start date
- Date of service falls after the member's coverage end date
- Member status was `terminated` as of the date of service
