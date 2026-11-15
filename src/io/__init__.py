"""I/O package: field contracts, loaders and the channel registry."""
from .channels import ChannelName, ChannelRegistry, ChannelState, ChannelStatus, probe_channels
from .loader import (
    ParsedFrameStats,
    canonicalise_categories,
    load_gtrends,
    load_price,
    load_restocks,
    load_sales,
    load_weather,
)
from .schema import (
    CATEGORICAL,
    TRENDS_CONTRACT,
    KEY_COLUMNS,
    PRICE_CONTRACT,
    RESTOCK_CONTRACT,
    SALES_CONTRACT,
    WEATHER_CONTRACT,
    WEATHER_FIELDS,
    ChannelContract,
    WeekColumnMap,
)

__all__ = [
    "CATEGORICAL", "ChannelContract", "ChannelName", "ChannelRegistry",
    "ChannelState", "ChannelStatus", "TRENDS_CONTRACT", "KEY_COLUMNS",
    "PRICE_CONTRACT", "ParsedFrameStats", "RESTOCK_CONTRACT", "SALES_CONTRACT",
    "WEATHER_CONTRACT", "WEATHER_FIELDS", "WeekColumnMap",
    "canonicalise_categories", "load_gtrends", "load_price", "load_restocks",
    "load_sales", "load_weather", "probe_channels",
]