# src/anchors.py
"""
Task 3, step 2 -- anchor sentences that define the FX channel axes.

Each channel is a direction in embedding space:

    axis = mean(embed(positive anchors)) - mean(embed(negative anchors))

and every headline is scored by its cosine similarity to that axis (SemAxis,
An, Kwak and Ahn, 2018). The POSITIVE pole is always the side that, by
standard FX theory, pushes USD/IDR UP (dollar stronger, rupiah weaker), so a
positive score means "this headline leans USD-bullish through this channel".

HOW THE ANCHORS WERE WRITTEN
---------------------------
  * From FX theory only, BEFORE any feature was compared with returns. They
    are frozen; changing them after looking at validation or test results
    would turn them into a tuned hyperparameter.
  * As MINIMAL PAIRS: the i-th positive anchor and the i-th negative anchor
    describe the same kind of event and differ only in direction ("Fed signals
    more rate hikes ahead" / "Fed signals rate cuts ahead"). Shared topic
    content therefore cancels in the centroid difference and the axis keeps
    only the directional contrast.
  * In wire-service headline style, to match the corpus.
  * Generic rather than event-specific: no dates, few named people, so an axis
    does not quietly encode one historical episode.
  * Ten per pole, so the robustness check can split them into two
    independent halves of five (split-half reliability, semantic_axes.py).

THE SEVEN CHANNELS
------------------
dollar  Global dollar strength. Our GDELT snapshot (19:00 WIB) is taken after
        the day's fixing, so these headlines report dollar moves in the
        European/US sessions that lagged JISDOR features cannot yet see.
risk    Risk-off vs risk-on. The dollar is the global safe haven, and
        emerging-market currencies are sold when risk appetite falls.
fed     US monetary stance. Higher expected US rates widen the rate
        differential in the dollar's favour (uncovered interest parity).
geo     Geopolitical escalation vs de-escalation. Conflict raises uncertainty
        and safe-haven demand.
oil     Oil price shocks. Indonesia is a net oil importer, so dearer oil
        widens the current-account deficit and weighs on the rupiah.
        (Caveat for the report: Indonesia exports coal and palm oil, so a
        broad commodity boom can also support the rupiah.)
em      Emerging-market capital flows. Portfolio outflows hit the rupiah
        directly; includes a few Indonesia-specific anchors.
trade   Trade-war escalation vs de-escalation. Tariffs hit Asian exporters and
        their currencies.
"""

from __future__ import annotations

import hashlib
import json

ANCHORS: dict[str, dict[str, list[str]]] = {
    "dollar": {
        "pos": [
            "Dollar climbs to multi-month high against major currencies",
            "US dollar strengthens as Treasury yields rise",
            "Greenback rallies broadly after strong US jobs data",
            "Dollar index extends gains as investors seek safety",
            "Dollar firms against Asian currencies",
            "US dollar posts biggest weekly gain in months",
            "Strong US economic data lifts the dollar",
            "Dollar surges as traders bet on higher US rates",
            "Euro and yen slide as dollar rebounds",
            "Dollar holds near peak on robust US growth",
        ],
        "neg": [
            "Dollar falls to multi-month low against major currencies",
            "US dollar weakens as Treasury yields drop",
            "Greenback slides broadly after weak US jobs data",
            "Dollar index extends losses as risk appetite improves",
            "Dollar eases against Asian currencies",
            "US dollar posts biggest weekly loss in months",
            "Soft US economic data weighs on the dollar",
            "Dollar slumps as traders bet on lower US rates",
            "Euro and yen climb as dollar weakens",
            "Dollar sinks to lowest level this year on slowing US growth",
        ],
    },
    "risk": {
        "pos": [
            "Global stocks tumble as investors flee to safe havens",
            "Markets sell off on fears of a global recession",
            "Investors dump risky assets amid market turmoil",
            "Volatility spikes as panic grips financial markets",
            "Gold hits record as investors seek safety",
            "Wall Street plunges in broad sell-off",
            "Asian shares slump as risk aversion deepens",
            "Credit markets seize up as fears of financial crisis grow",
            "Investors rush into government bonds as stocks crash",
            "Global markets rattled by banking sector fears",
        ],
        "neg": [
            "Global stocks rally as investors return to risky assets",
            "Markets climb as recession fears ease",
            "Investors pile into risky assets as markets calm",
            "Volatility falls as calm returns to financial markets",
            "Gold slips as investors move back into equities",
            "Wall Street surges in broad rally",
            "Asian shares jump as risk appetite improves",
            "Credit markets steady as financial crisis fears fade",
            "Investors sell safe-haven bonds as stocks rebound",
            "Global markets cheered by strong bank earnings",
        ],
    },
    "fed": {
        "pos": [
            "Fed raises interest rates to fight inflation",
            "Fed signals more rate hikes ahead",
            "Hot US inflation data boosts bets on further Fed tightening",
            "Powell says Fed will keep rates higher for longer",
            "Fed officials push back against expectations of rate cuts",
            "Treasury yields jump as Fed takes hawkish stance",
            "Fed minutes show policymakers favour further tightening",
            "Strong US jobs report raises pressure on Fed to hike",
            "Fed delays rate cuts as inflation proves sticky",
            "Markets price in higher US interest rates",
        ],
        "neg": [
            "Fed cuts interest rates to support the economy",
            "Fed signals rate cuts ahead",
            "Cooling US inflation data boosts bets on Fed easing",
            "Powell says Fed is ready to lower rates",
            "Fed officials open the door to rate cuts",
            "Treasury yields fall as Fed takes dovish stance",
            "Fed minutes show policymakers favour easing",
            "Weak US jobs report raises pressure on Fed to cut",
            "Fed brings forward rate cuts as inflation eases",
            "Markets price in lower US interest rates",
        ],
    },
    "geo": {
        "pos": [
            "Missile strikes escalate conflict, raising fears of wider war",
            "Troops mass at the border as tensions soar",
            "Fighting intensifies as peace talks collapse",
            "Military launches major offensive against neighbouring country",
            "Sanctions tightened as standoff between powers deepens",
            "Warships deployed as regional tensions escalate",
            "Attack on shipping route sparks fears of wider conflict",
            "Nuclear threats raise alarm over escalating crisis",
            "Ceasefire collapses as hostilities resume",
            "Superpowers trade threats as diplomatic crisis worsens",
        ],
        "neg": [
            "Ceasefire agreed, easing fears of wider war",
            "Troops withdraw from the border as tensions ease",
            "Peace talks make progress as fighting subsides",
            "Military halts offensive and agrees to negotiate",
            "Sanctions eased as relations between powers thaw",
            "Warships recalled as regional tensions cool",
            "Shipping route reopens as conflict fears fade",
            "Nuclear talks raise hopes of ending crisis",
            "Ceasefire holds as hostilities end",
            "Leaders hold summit as diplomatic relations improve",
        ],
    },
    "oil": {
        "pos": [
            "Oil prices surge on supply disruption fears",
            "Crude jumps as OPEC+ announces deep output cuts",
            "Brent climbs above $100 a barrel",
            "Oil rallies as Middle East tensions threaten supplies",
            "Fuel prices soar as energy crunch deepens",
            "Crude hits multi-month high on tight supply",
            "Oil spikes after attack on energy facilities",
            "Energy prices climb as inventories fall sharply",
            "Oil extends gains on strong demand outlook",
            "Gas and oil prices jump amid supply crisis",
        ],
        "neg": [
            "Oil prices slump on oversupply fears",
            "Crude falls as OPEC+ boosts output",
            "Brent drops below $60 a barrel",
            "Oil slides as Middle East tensions ease",
            "Fuel prices fall as energy crunch eases",
            "Crude hits multi-month low on supply glut",
            "Oil tumbles as energy supplies recover",
            "Energy prices drop as inventories build",
            "Oil extends losses on weak demand outlook",
            "Gas and oil prices plunge as supply crisis eases",
        ],
    },
    "em": {
        "pos": [
            "Foreign investors pull money out of emerging markets",
            "Emerging market currencies slide as capital flees",
            "Asian currencies tumble on heavy foreign outflows",
            "Rupiah weakens as foreigners sell Indonesian bonds",
            "Emerging market stocks slump as investors retreat",
            "Central bank intervenes to defend falling currency",
            "Capital outflows deepen emerging market turmoil",
            "Investors flee developing economies amid dollar strength",
            "Emerging market debt sell-off accelerates",
            "Indonesian stocks fall as foreign selling intensifies",
        ],
        "neg": [
            "Foreign investors pour money into emerging markets",
            "Emerging market currencies rally on capital inflows",
            "Asian currencies climb on strong foreign inflows",
            "Rupiah strengthens as foreigners buy Indonesian bonds",
            "Emerging market stocks rally as investors return",
            "Currency rebounds, easing pressure on central bank",
            "Capital inflows lift emerging market assets",
            "Investors return to developing economies as dollar weakens",
            "Emerging market debt rally gathers pace",
            "Indonesian stocks rise as foreign buying picks up",
        ],
    },
    "trade": {
        "pos": [
            "US imposes sweeping new tariffs on imports",
            "Trade war escalates as China retaliates with tariffs",
            "Talks collapse as trade tensions intensify",
            "New export controls deepen trade dispute",
            "Tariff threats rattle global supply chains",
            "Countries hit back with retaliatory tariffs",
            "Trade barriers rise as protectionism spreads",
            "Washington threatens tariffs on allies",
            "Global trade slows as tariffs bite",
            "Trade dispute deepens with fresh restrictions on exports",
        ],
        "neg": [
            "US lifts tariffs on imports",
            "Trade war eases as China and US reach deal",
            "Talks succeed as trade tensions ease",
            "Export controls relaxed in trade dispute",
            "Tariff truce calms global supply chains",
            "Countries agree to scrap retaliatory tariffs",
            "Trade barriers fall as free trade deal signed",
            "Washington exempts allies from tariffs",
            "Global trade rebounds as tariffs are removed",
            "Trade dispute resolved with new agreement",
        ],
    },
}

CHANNELS = tuple(ANCHORS)


def anchors_hash() -> str:
    """Fingerprint of the anchor text; invalidates cached anchor embeddings."""
    blob = json.dumps(ANCHORS, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _check() -> None:
    for ch, poles in ANCHORS.items():
        assert len(poles["pos"]) == len(poles["neg"]) == 10, ch
        assert not set(poles["pos"]) & set(poles["neg"]), ch


_check()
