"""MOSAIC API Request and Response Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SupportRequest(BaseModel):
    """Customer support request payload submitted to the Intake API endpoint."""
    customer_id: str = Field(..., example="cust_101", description="Unique customer identifier")
    raw_message: str = Field(..., example="I need a refund for pay_101 right away.", description="Raw text of customer support message")
    initial_facts: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        example={"identity_verified": True, "payment_verified": True},
        description="Key-value state assertions (facts) required for validation"
    )
    active_flags: Optional[List[str]] = Field(
        default_factory=list,
        example=[],
        description="Active policy flags (e.g. ['SUSPICIOUS_LOCATION_LOGIN'])"
    )
