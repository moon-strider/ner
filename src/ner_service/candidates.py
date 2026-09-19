from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CANDIDATE_GENERATOR_VERSION: int = 1

_PUNCT_CATS = ("P", "S", "Z", "C")
_EDGE_KEEP = "+#"
_FORM_PUNCT = "-./@:_"
_NUMISH = frozenset("0123456789.,:/-+=%$")
_TOKEN_RE = re.compile(r"\S+")
_MAX_BRIDGE = 3
_MAX_VARIANT_LEN = 8
_MAX_FULL_VARIANT_SPAN = 12

_PRIORITY_FORM = 0
_PRIORITY_CAPITALIZED = 1
_PRIORITY_RARE = 2
_PRIORITY_REPEAT = 3
_PRIORITY_JOINED = 4

_SOURCE_BY_PRIORITY = {
    _PRIORITY_FORM: "form",
    _PRIORITY_CAPITALIZED: "capitalized",
    _PRIORITY_RARE: "rare",
    _PRIORITY_REPEAT: "repeat",
    _PRIORITY_JOINED: "joined",
}


def _words(blob: str) -> frozenset[str]:
    return frozenset(re.findall(r"\S+", blob))


STOP_EN: frozenset[str] = _words(
    "the a an and or but if of to in on at by for with from into onto over under after before "
    "during between against about as than that this these those it its he she they them him his "
    "her their theirs we us our you your i me my mine myself is are was were be been being has "
    "have had having do does did done will would can could shall should may might must not no "
    "nor so up down out off again also then when where why how which who whom whose there here "
    "now today yesterday tomorrow last next week month year day time more most less least very "
    "just only all any both each few other some such own same too s t re ve ll d m new said "
    "says say one two three four five six seven eight nine ten first second third because while "
    "since until though although whether either neither per via yet still even ever never "
    "always often sometimes usually once twice many much several every another any each both "
    "either neither make makes made get gets got go goes went come comes came take takes took "
    "give gives gave see sees saw know knows knew think thinks thought want wants wanted need "
    "needs needed use uses used find finds found tell tells told ask asks asked work works "
    "worked seem seems seemed feel feels felt try tries tried leave leaves left call calls "
    "called keep keeps kept let lets put puts set sets run runs ran move moves moved live lives "
    "lived believe believes believed hold holds held bring brings brought happen happens "
    "happened write writes wrote provide provides provided sit sits sat stand stands stood lose "
    "loses lost pay pays paid meet meets met include includes included continue continues "
    "continued learn learns learnt change changes changed lead leads led understand understands "
    "understood watch watches watched follow follows followed stop stops stopped create creates "
    "created speak speaks spoke read reads allow allows allowed add adds added spend spends "
    "spent grow grows grew open opens opened walk walks walked win wins won offer offers "
    "offered remember remembers remembered love loves loved consider considers considered "
    "appear appears appeared buy buys bought wait waits waited serve serves served die dies "
    "died send sends sent expect expects expected build builds built stay stays stayed fall "
    "falls fell cut cuts reach reaches reached kill kills killed remain remains remained "
    "suggest suggests suggested raise raises raised pass passes passed sell sells sold require "
    "requires required report reports reported decide decides decided pull pulls pulled "
)

STOP_RU: frozenset[str] = _words(
    "\u0438 \u0432 \u0432\u043e \u043d\u0435 \u0447\u0442\u043e \u043e\u043d \u043d\u0430 "
    "\u044f \u0441 \u0441\u043e \u043a\u0430\u043a \u0430 \u0442\u043e \u0432\u0441\u0435 "
    "\u043e\u043d\u0430 \u0442\u0430\u043a \u0435\u0433\u043e \u043d\u043e \u0434\u0430 "
    "\u0442\u044b \u043a \u0443 \u0436\u0435 \u0432\u044b \u0437\u0430 \u0431\u044b "
    "\u043f\u043e \u0442\u043e\u043b\u044c\u043a\u043e \u0435\u0435 \u043c\u043d\u0435 "
    "\u0431\u044b\u043b\u043e \u0432\u043e\u0442 \u043e\u0442 \u043c\u0435\u043d\u044f "
    "\u0435\u0449\u0435 \u043d\u0435\u0442 \u043e \u0438\u0437 \u0435\u043c\u0443 "
    "\u0442\u0435\u043f\u0435\u0440\u044c \u043a\u043e\u0433\u0434\u0430 "
    "\u0434\u0430\u0436\u0435 \u043d\u0443 \u0432\u0434\u0440\u0443\u0433 \u043b\u0438 "
    "\u0435\u0441\u043b\u0438 \u0443\u0436\u0435 \u0438\u043b\u0438 \u043d\u0438 "
    "\u0431\u044b\u0442\u044c \u0431\u044b\u043b \u043d\u0435\u0433\u043e \u0434\u043e "
    "\u0432\u0430\u0441 \u043e\u043f\u044f\u0442\u044c \u0443\u0436 \u0432\u0430\u043c "
    "\u0432\u0435\u0434\u044c \u0442\u0430\u043c \u043f\u043e\u0442\u043e\u043c "
    "\u0441\u0435\u0431\u044f \u043d\u0438\u0447\u0435\u0433\u043e \u0435\u0439 "
    "\u043c\u043e\u0436\u0435\u0442 \u043e\u043d\u0438 \u0442\u0443\u0442 \u0433\u0434\u0435 "
    "\u0435\u0441\u0442\u044c \u043d\u0430\u0434\u043e \u043d\u0435\u0439 \u0434\u043b\u044f "
    "\u043c\u044b \u0442\u0435\u0431\u044f \u0438\u0445 \u0447\u0435\u043c "
    "\u0431\u044b\u043b\u0430 \u0441\u0430\u043c \u0447\u0442\u043e\u0431 \u0431\u0435\u0437 "
    "\u0431\u0443\u0434\u0442\u043e \u0447\u0435\u0433\u043e \u0440\u0430\u0437 "
    "\u0442\u043e\u0436\u0435 \u0441\u0435\u0431\u0435 \u043f\u043e\u0434 "
    "\u0431\u0443\u0434\u0435\u0442 \u0436 \u0442\u043e\u0433\u0434\u0430 \u043a\u0442\u043e "
    "\u044d\u0442\u043e\u0442 \u0442\u043e\u0433\u043e \u043f\u043e\u0442\u043e\u043c\u0443 "
    "\u044d\u0442\u043e\u0433\u043e \u043a\u0430\u043a\u043e\u0439 "
    "\u0441\u043e\u0432\u0441\u0435\u043c \u043d\u0438\u043c \u0437\u0434\u0435\u0441\u044c "
    "\u044d\u0442\u043e\u043c \u043e\u0434\u0438\u043d \u043f\u043e\u0447\u0442\u0438 "
    "\u043c\u043e\u0439 \u0442\u0435\u043c \u0447\u0442\u043e\u0431\u044b \u043d\u0435\u0435 "
    "\u0441\u0435\u0439\u0447\u0430\u0441 \u0431\u044b\u043b\u0438 \u043a\u0443\u0434\u0430 "
    "\u0437\u0430\u0447\u0435\u043c \u0432\u0441\u0435\u0445 "
    "\u043d\u0438\u043a\u043e\u0433\u0434\u0430 \u043c\u043e\u0436\u043d\u043e "
    "\u043f\u0440\u0438 \u043d\u0430\u043a\u043e\u043d\u0435\u0446 \u0434\u0432\u0430 "
    "\u043e\u0431 \u0434\u0440\u0443\u0433\u043e\u0439 \u0445\u043e\u0442\u044c "
    "\u043f\u043e\u0441\u043b\u0435 \u043d\u0430\u0434 \u0431\u043e\u043b\u044c\u0448\u0435 "
    "\u0442\u043e\u0442 \u0447\u0435\u0440\u0435\u0437 \u044d\u0442\u0438 \u043d\u0430\u0441 "
    "\u043f\u0440\u043e \u0432\u0441\u0435\u0433\u043e \u043d\u0438\u0445 "
    "\u043a\u0430\u043a\u0430\u044f \u043c\u043d\u043e\u0433\u043e "
    "\u0440\u0430\u0437\u0432\u0435 \u0442\u0440\u0438 \u044d\u0442\u0443 \u043c\u043e\u044f "
    "\u0432\u043f\u0440\u043e\u0447\u0435\u043c \u0445\u043e\u0440\u043e\u0448\u043e "
    "\u0441\u0432\u043e\u044e \u044d\u0442\u043e\u0439 \u043f\u0435\u0440\u0435\u0434 "
    "\u0438\u043d\u043e\u0433\u0434\u0430 \u043b\u0443\u0447\u0448\u0435 "
    "\u0447\u0443\u0442\u044c \u0442\u043e\u043c \u043d\u0435\u043b\u044c\u0437\u044f "
    "\u0442\u0430\u043a\u043e\u0439 \u0438\u043c \u0431\u043e\u043b\u0435\u0435 "
    "\u0432\u0441\u0435\u0433\u0434\u0430 \u043a\u043e\u043d\u0435\u0447\u043d\u043e "
    "\u0432\u0441\u044e \u043c\u0435\u0436\u0434\u0443 \u044d\u0442\u043e "
    "\u0437\u043d\u0430\u0447\u0438\u0442 \u0432\u0440\u043e\u0434\u0435 "
    "\u0432\u0435\u0441\u044c \u043a\u043e\u0433\u043e \u0437\u0430\u0442\u043e "
    "\u0441\u0432\u043e\u0439 \u043d\u0430\u0448 \u043a\u0442\u043e-\u0442\u043e "
    "\u044d\u0442\u043e\u0442 \u044d\u0442\u0430 \u044d\u0442\u0438 \u043e\u043d\u043e "
    "\u043c\u044b \u0432\u044b \u043e\u043d\u0438 \u044f \u0442\u044b \u0441\u0435\u0431\u044f "
    "\u0432\u0435\u0441\u044c \u0432\u0441\u044f \u0432\u0441\u0451 \u0432\u0441\u0435 "
    "\u0441\u0430\u043c \u0441\u0430\u043c\u0430 \u0441\u0430\u043c\u043e "
    "\u0441\u0430\u043c\u0438 \u043e\u0434\u0438\u043d \u043e\u0434\u043d\u0430 "
    "\u043e\u0434\u043d\u043e \u043e\u0434\u043d\u0438 \u0434\u0432\u0430 \u0434\u0432\u0435 "
    "\u0442\u0440\u0438 \u0447\u0435\u0442\u044b\u0440\u0435 \u043f\u044f\u0442\u044c "
    "\u0448\u0435\u0441\u0442\u044c \u0441\u0435\u043c\u044c "
    "\u0432\u043e\u0441\u0435\u043c\u044c \u0434\u0435\u0432\u044f\u0442\u044c "
    "\u0434\u0435\u0441\u044f\u0442\u044c \u043f\u0435\u0440\u0432\u044b\u0439 "
    "\u0432\u0442\u043e\u0440\u043e\u0439 \u0442\u0440\u0435\u0442\u0438\u0439 "
    "\u0431\u044b\u043b \u0431\u044b\u043b\u0430 \u0431\u044b\u043b\u043e "
    "\u0431\u044b\u043b\u0438 \u0431\u0443\u0434\u0443 \u0431\u0443\u0434\u0435\u0448\u044c "
    "\u0431\u0443\u0434\u0435\u0442 \u0431\u0443\u0434\u0435\u043c "
    "\u0431\u0443\u0434\u0435\u0442\u0435 \u0431\u0443\u0434\u0443\u0442 "
    "\u0435\u0441\u0442\u044c \u0434\u0435\u043b\u0430\u0442\u044c "
    "\u0434\u0435\u043b\u0430\u043b \u0434\u0435\u043b\u0430\u0435\u0442 "
    "\u0434\u0435\u043b\u0430\u044e\u0442 \u0441\u0434\u0435\u043b\u0430\u0442\u044c "
    "\u0441\u0434\u0435\u043b\u0430\u043b \u0441\u0434\u0435\u043b\u0430\u043b\u0430 "
    "\u0433\u043e\u0432\u043e\u0440\u0438\u0442\u044c "
    "\u0433\u043e\u0432\u043e\u0440\u0438\u0442 \u0433\u043e\u0432\u043e\u0440\u0438\u043b "
    "\u0441\u043a\u0430\u0437\u0430\u0442\u044c \u0441\u043a\u0430\u0437\u0430\u043b "
    "\u0441\u043a\u0430\u0437\u0430\u0442\u044c \u0437\u043d\u0430\u0442\u044c "
    "\u0437\u043d\u0430\u0435\u0442 \u0437\u043d\u0430\u043b \u0437\u043d\u0430\u0442\u044c "
    "\u0443\u043c\u0435\u0442\u044c \u043c\u043e\u0436\u0435\u0442 "
    "\u043c\u043e\u0433\u0443\u0442 \u043c\u043e\u0433 \u0445\u043e\u0442\u0435\u0442\u044c "
    "\u0445\u043e\u0447\u0435\u0442 \u0445\u043e\u0442\u0435\u043b "
    "\u0445\u043e\u0442\u0435\u0442\u044c \u043d\u0430\u0434\u043e "
    "\u043d\u0443\u0436\u043d\u043e \u043d\u0443\u0436\u0435\u043d "
    "\u043d\u0443\u0436\u043d\u0430 \u0438\u0434\u0442\u0438 \u0438\u0434\u0435\u0442 "
    "\u0448\u0435\u043b \u043f\u0440\u0438\u0439\u0442\u0438 "
    "\u043f\u0440\u0438\u0448\u0435\u043b \u0434\u0430\u0442\u044c \u0434\u0430\u0435\u0442 "
    "\u0434\u0430\u043b \u0432\u0437\u044f\u0442\u044c \u0432\u0437\u044f\u043b "
    "\u0432\u0438\u0434\u0435\u0442\u044c \u0432\u0438\u0434\u0438\u0442 "
    "\u0432\u0438\u0434\u0435\u043b \u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c "
    "\u0441\u043c\u043e\u0442\u0440\u0438\u0442 \u0441\u043c\u043e\u0442\u0440\u0435\u043b "
    "\u0440\u0430\u0431\u043e\u0442\u0430\u0442\u044c "
    "\u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 "
    "\u0440\u0430\u0431\u043e\u0442\u0430\u043b "
    "\u0440\u0430\u0431\u043e\u0442\u0430\u0442\u044c "
    "\u0441\u0434\u0435\u043b\u0430\u0442\u044c \u0434\u0435\u043b\u0430\u0442\u044c "
    "\u0434\u0430\u0442\u044c \u0434\u0430\u0439 \u0434\u0430\u0439\u0442\u0435 "
    "\u0432\u043e\u0437\u044c\u043c\u0438 \u0441\u043a\u0438\u043d\u044c "
    "\u0441\u043a\u0438\u043d\u044c\u0442\u0435 "
    "\u043f\u0435\u0440\u0435\u043a\u0438\u043d\u044c "
    "\u043f\u0435\u0440\u0435\u0432\u0435\u0434\u0438 \u043d\u0430\u043f\u0438\u0448\u0438 "
    "\u043d\u0430\u043f\u0438\u0448\u0438 \u0441\u043a\u0430\u0436\u0438 "
    "\u043f\u043e\u0437\u0432\u043e\u043d\u0438 \u043e\u043f\u043b\u0430\u0442\u0438 "
    "\u043e\u0442\u043f\u0440\u0430\u0432\u044c \u0437\u0430\u043a\u0430\u0436\u0438 "
    "\u043a\u0443\u043f\u0438 \u0441\u043e\u0437\u0434\u0430\u0439 "
    "\u043f\u0440\u043e\u0434\u043b\u0438 \u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0438 "
    "\u043f\u0440\u043e\u0432\u0435\u0440\u044c "
)

_FREQUENT_EN: frozenset[str] = _words(
    "man woman child boy girl people person world life hand part place case group company "
    "number way system program question work government night point home water room mother area "
    "money story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office door "
    "health person art war history party result change morning reason research girl guy moment "
    "air teacher force education foot boy age policy process music market sense nation plan "
    "college interest death experience effect use class control care field development role "
    "effort rate heart drug show leader light voice wife police mind price report decision son "
    "view relationship town road arm difference value building action model season society tax "
    "director position player record paper space ground form event official matter center "
    "couple site project activity star table need court production eat American teach oil "
    "situation cost industry figure street image phone data computer language file code page "
    "text test version release package library server client request response error message "
    "config configuration option default value string list object function method class module "
    "import export return result output input build deploy run time token model prompt state "
    "question answer label entity type word name span line document sentence threshold "
    "probability score precision recall cost budget cache window script source target user "
    "account access key secret token file directory path format text "
)

_FREQUENT_RU: frozenset[str] = _words(
    "\u0447\u0435\u043b\u043e\u0432\u0435\u043a \u043b\u044e\u0434\u0438 \u043c\u0438\u0440 "
    "\u0436\u0438\u0437\u043d\u044c \u0440\u0443\u043a\u0430 \u0447\u0430\u0441\u0442\u044c "
    "\u043c\u0435\u0441\u0442\u043e \u0441\u043b\u0443\u0447\u0430\u0439 "
    "\u0433\u0440\u0443\u043f\u043f\u0430 \u043a\u043e\u043c\u043f\u0430\u043d\u0438\u044f "
    "\u043d\u043e\u043c\u0435\u0440 \u0441\u043f\u043e\u0441\u043e\u0431 "
    "\u0441\u0438\u0441\u0442\u0435\u043c\u0430 "
    "\u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 "
    "\u0432\u043e\u043f\u0440\u043e\u0441 \u0440\u0430\u0431\u043e\u0442\u0430 "
    "\u043f\u0440\u0430\u0432\u0438\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u043e "
    "\u043d\u043e\u0447\u044c \u0442\u043e\u0447\u043a\u0430 \u0434\u043e\u043c "
    "\u0432\u043e\u0434\u0430 \u043a\u043e\u043c\u043d\u0430\u0442\u0430 "
    "\u043c\u0430\u043c\u0430 \u043e\u0431\u043b\u0430\u0441\u0442\u044c "
    "\u0434\u0435\u043d\u044c\u0433\u0438 \u0438\u0441\u0442\u043e\u0440\u0438\u044f "
    "\u0444\u0430\u043a\u0442 \u043c\u0435\u0441\u044f\u0446 \u043f\u0440\u0430\u0432\u043e "
    "\u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435 "
    "\u043a\u043d\u0438\u0433\u0430 \u0433\u043b\u0430\u0437 \u0434\u0435\u043b\u043e "
    "\u0441\u043b\u043e\u0432\u043e \u0431\u0438\u0437\u043d\u0435\u0441 "
    "\u0441\u0442\u043e\u0440\u043e\u043d\u0430 \u0432\u0438\u0434 "
    "\u0433\u043e\u043b\u043e\u0432\u0430 \u0434\u043e\u043c "
    "\u0441\u0435\u0440\u0432\u0438\u0441 \u0434\u0440\u0443\u0433 \u043e\u0442\u0435\u0446 "
    "\u0432\u043b\u0430\u0441\u0442\u044c \u0447\u0430\u0441 \u0438\u0433\u0440\u0430 "
    "\u043b\u0438\u043d\u0438\u044f \u043a\u043e\u043d\u0435\u0446 \u0447\u043b\u0435\u043d "
    "\u0437\u0430\u043a\u043e\u043d \u043c\u0430\u0448\u0438\u043d\u0430 "
    "\u0433\u043e\u0440\u043e\u0434 "
    "\u0441\u043e\u043e\u0431\u0449\u0435\u0441\u0442\u0432\u043e \u0438\u043c\u044f "
    "\u043f\u0440\u0435\u0437\u0438\u0434\u0435\u043d\u0442 "
    "\u043a\u043e\u043c\u0430\u043d\u0434\u0430 \u043c\u0438\u043d\u0443\u0442\u0430 "
    "\u0438\u0434\u0435\u044f \u0442\u0435\u043b\u043e "
    "\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044f "
    "\u0440\u043e\u0434\u0438\u0442\u0435\u043b\u044c \u043b\u0438\u0446\u043e "
    "\u043e\u0444\u0438\u0441 \u0434\u0432\u0435\u0440\u044c "
    "\u0437\u0434\u043e\u0440\u043e\u0432\u044c\u0435 "
    "\u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u043e \u0432\u043e\u0439\u043d\u0430 "
    "\u0438\u0441\u0442\u043e\u0440\u0438\u044f "
    "\u0432\u0435\u0447\u0435\u0440\u0438\u043d\u043a\u0430 "
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 "
    "\u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0443\u0442\u0440\u043e "
    "\u043f\u0440\u0438\u0447\u0438\u043d\u0430 \u0432\u043e\u0437\u0434\u0443\u0445 "
    "\u0443\u0447\u0438\u0442\u0435\u043b\u044c \u0441\u0438\u043b\u0430 "
    "\u043e\u0431\u0440\u0430\u0437\u043e\u0432\u0430\u043d\u0438\u0435 "
    "\u043d\u043e\u0433\u0430 \u0432\u043e\u0437\u0440\u0430\u0441\u0442 "
    "\u043f\u043e\u043b\u0438\u0442\u0438\u043a\u0430 "
    "\u043f\u0440\u043e\u0446\u0435\u0441\u0441 \u043c\u0443\u0437\u044b\u043a\u0430 "
    "\u0440\u044b\u043d\u043e\u043a \u0441\u043c\u044b\u0441\u043b "
    "\u043d\u0430\u0446\u0438\u044f \u043f\u043b\u0430\u043d "
    "\u043a\u043e\u043b\u043b\u0435\u0434\u0436 \u0438\u043d\u0442\u0435\u0440\u0435\u0441 "
    "\u0441\u043c\u0435\u0440\u0442\u044c \u043e\u043f\u044b\u0442 "
    "\u044d\u0444\u0444\u0435\u043a\u0442 \u043a\u043b\u0430\u0441\u0441 "
    "\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c \u0437\u0430\u0431\u043e\u0442\u0430 "
    "\u043e\u0431\u043b\u0430\u0441\u0442\u044c "
    "\u0440\u0430\u0437\u0432\u0438\u0442\u0438\u0435 \u0440\u043e\u043b\u044c "
    "\u0443\u0441\u0438\u043b\u0438\u0435 \u0441\u043a\u043e\u0440\u043e\u0441\u0442\u044c "
    "\u0441\u0435\u0440\u0434\u0446\u0435 "
    "\u043b\u0435\u043a\u0430\u0440\u0441\u0442\u0432\u043e \u0448\u043e\u0443 "
    "\u043b\u0438\u0434\u0435\u0440 \u0441\u0432\u0435\u0442 \u0433\u043e\u043b\u043e\u0441 "
    "\u0436\u0435\u043d\u0430 \u043f\u043e\u043b\u0438\u0446\u0438\u044f \u0443\u043c "
    "\u0446\u0435\u043d\u0430 \u043e\u0442\u0447\u0435\u0442 "
    "\u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u0441\u044b\u043d \u0432\u0438\u0434 "
    "\u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u0435 \u0433\u043e\u0440\u043e\u0434 "
    "\u0434\u043e\u0440\u043e\u0433\u0430 \u0440\u0443\u043a\u0430 "
    "\u0440\u0430\u0437\u043d\u0438\u0446\u0430 "
    "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u0437\u0434\u0430\u043d\u0438\u0435 "
    "\u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435 \u043c\u043e\u0434\u0435\u043b\u044c "
    "\u0441\u0435\u0437\u043e\u043d \u043e\u0431\u0449\u0435\u0441\u0442\u0432\u043e "
    "\u043d\u0430\u043b\u043e\u0433 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440 "
    "\u043f\u043e\u0437\u0438\u0446\u0438\u044f \u0438\u0433\u0440\u043e\u043a "
    "\u0437\u0430\u043f\u0438\u0441\u044c \u0431\u0443\u043c\u0430\u0433\u0430 "
    "\u043f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u043e "
    "\u0437\u0435\u043c\u043b\u044f \u0444\u043e\u0440\u043c\u0430 "
    "\u0441\u043e\u0431\u044b\u0442\u0438\u0435 "
    "\u0447\u0438\u043d\u043e\u0432\u043d\u0438\u043a \u0432\u043e\u043f\u0440\u043e\u0441 "
    "\u0446\u0435\u043d\u0442\u0440 \u043f\u0430\u0440\u0430 \u0441\u0430\u0439\u0442 "
    "\u043f\u0440\u043e\u0435\u043a\u0442 "
    "\u0434\u0435\u044f\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044c "
    "\u0437\u0432\u0435\u0437\u0434\u0430 \u0442\u0430\u0431\u043b\u0438\u0446\u0430 "
    "\u043f\u043e\u0442\u0440\u0435\u0431\u043d\u043e\u0441\u0442\u044c \u0441\u0443\u0434 "
    "\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0441\u0442\u0432\u043e "
    "\u0435\u0441\u0442\u044c \u043d\u0435\u0444\u0442\u044c "
    "\u0441\u0438\u0442\u0443\u0430\u0446\u0438\u044f "
    "\u043e\u0442\u0440\u0430\u0441\u043b\u044c \u0444\u0438\u0433\u0443\u0440\u0430 "
    "\u0443\u043b\u0438\u0446\u0430 \u043e\u0431\u0440\u0430\u0437 "
    "\u0442\u0435\u043b\u0435\u0444\u043e\u043d \u0434\u0430\u043d\u043d\u044b\u0435 "
    "\u043a\u043e\u043c\u043f\u044c\u044e\u0442\u0435\u0440 \u044f\u0437\u044b\u043a "
    "\u0444\u0430\u0439\u043b \u043a\u043e\u0434 "
    "\u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u0442\u0435\u043a\u0441\u0442 "
    "\u0442\u0435\u0441\u0442 \u0432\u0435\u0440\u0441\u0438\u044f "
    "\u0440\u0435\u043b\u0438\u0437 \u043f\u0430\u043a\u0435\u0442 "
    "\u0431\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0430 "
    "\u0441\u0435\u0440\u0432\u0435\u0440 \u043a\u043b\u0438\u0435\u043d\u0442 "
    "\u0437\u0430\u043f\u0440\u043e\u0441 \u043e\u0442\u0432\u0435\u0442 "
    "\u043e\u0448\u0438\u0431\u043a\u0430 "
    "\u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 "
    "\u043a\u043e\u043d\u0444\u0438\u0433 "
    "\u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 \u043e\u043f\u0446\u0438\u044f "
    "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u0441\u0442\u0440\u043e\u043a\u0430 "
    "\u0441\u043f\u0438\u0441\u043e\u043a \u043e\u0431\u044a\u0435\u043a\u0442 "
    "\u0444\u0443\u043d\u043a\u0446\u0438\u044f \u043c\u0435\u0442\u043e\u0434 "
    "\u043a\u043b\u0430\u0441\u0441 \u043c\u043e\u0434\u0443\u043b\u044c "
    "\u0438\u043c\u043f\u043e\u0440\u0442 \u044d\u043a\u0441\u043f\u043e\u0440\u0442 "
    "\u0432\u043e\u0437\u0432\u0440\u0430\u0442 "
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0432\u044b\u0432\u043e\u0434 "
    "\u0432\u0445\u043e\u0434 \u0441\u0431\u043e\u0440\u043a\u0430 "
    "\u0434\u0435\u043f\u043b\u043e\u0439 \u0432\u0440\u0435\u043c\u044f "
    "\u0442\u043e\u043a\u0435\u043d \u043c\u043e\u0434\u0435\u043b\u044c "
    "\u043f\u0440\u043e\u043c\u043f\u0442 "
    "\u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 "
    "\u0432\u043e\u043f\u0440\u043e\u0441 \u043e\u0442\u0432\u0435\u0442 "
    "\u043c\u0435\u0442\u043a\u0430 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u044c "
    "\u0442\u0438\u043f \u0441\u043b\u043e\u0432\u043e \u0438\u043c\u044f "
    "\u0438\u043d\u0442\u0435\u0440\u0432\u0430\u043b "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 "
    "\u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435 "
    "\u043f\u043e\u0440\u043e\u0433 "
    "\u0432\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0441\u0442\u044c "
    "\u043e\u0446\u0435\u043d\u043a\u0430 \u0442\u043e\u0447\u043d\u043e\u0441\u0442\u044c "
    "\u043f\u043e\u043b\u043d\u043e\u0442\u0430 "
    "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c "
    "\u0431\u044e\u0434\u0436\u0435\u0442 \u043a\u044d\u0448 \u043e\u043a\u043d\u043e "
    "\u0441\u043a\u0440\u0438\u043f\u0442 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a "
    "\u0446\u0435\u043b\u044c "
    "\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c "
    "\u0430\u043a\u043a\u0430\u0443\u043d\u0442 \u0434\u043e\u0441\u0442\u0443\u043f "
    "\u043a\u043b\u044e\u0447 \u0441\u0435\u043a\u0440\u0435\u0442 \u0444\u0430\u0439\u043b "
    "\u043a\u0430\u0442\u0430\u043b\u043e\u0433 \u043f\u0443\u0442\u044c "
    "\u0444\u043e\u0440\u043c\u0430\u0442 "
)

COMMON: frozenset[str] = STOP_EN | STOP_RU | _FREQUENT_EN | _FREQUENT_RU

CONNECTORS: frozenset[str] = frozenset(
    {
        "of",
        "the",
        "and",
        "&",
        "de",
        "da",
        "van",
        "von",
        "del",
        "di",
        "la",
        "le",
        "el",
        "al",
        "bin",
        "ibn",
        "for",
        "in",
        "on",
        "at",
        "du",
        "des",
    }
)


@dataclass(frozen=True)
class Candidate:
    text: str
    start: int
    end: int
    source: str
    priority: int


@dataclass(frozen=True)
class _Token:
    core: str
    start: int
    end: int


def _is_edge_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in _PUNCT_CATS and ch not in _EDGE_KEEP


def _core_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and _is_edge_punct(text[start]):
        start += 1
    while end > start and _is_edge_punct(text[end - 1]):
        end -= 1
    return start, end


def _tokenize(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    for match in _TOKEN_RE.finditer(text):
        start, end = _core_bounds(text, match.start(), match.end())
        tokens.append(_Token(core=text[start:end], start=start, end=end))
    return tuple(tokens)


def _is_punct_only(core: str) -> bool:
    return core == "" or all(unicodedata.category(ch)[0] in _PUNCT_CATS for ch in core)


def _is_numeric(core: str) -> bool:
    return bool(core) and all(ch in _NUMISH for ch in core) and any(ch.isdigit() for ch in core)


def _is_content(core: str) -> bool:
    if len(core) < 2 or _is_punct_only(core) or _is_numeric(core):
        return False
    low = core.lower()
    return low not in STOP_EN and low not in STOP_RU


def _is_capitalized(core: str) -> bool:
    return bool(core) and core[0].isupper()


def _has_form(core: str) -> bool:
    return any(ch.isdigit() for ch in core) or any(ch in _FORM_PUNCT for ch in core[1:-1])


def _is_bridge(core: str) -> bool:
    if core == "" or _is_punct_only(core) or _is_numeric(core):
        return False
    low = core.lower()
    return low in STOP_EN or low in STOP_RU or low in CONNECTORS


def _repeat_indices(tokens: tuple[_Token, ...]) -> frozenset[int]:
    seq: list[tuple[int, str]] = [(i, tok.core.lower()) for i, tok in enumerate(tokens) if tok.core]
    grams: dict[tuple[str, ...], int] = {}
    for size in (1, 2, 3, 4):
        for pos in range(len(seq) - size + 1):
            gram = tuple(word for _index, word in seq[pos : pos + size])
            grams[gram] = grams.get(gram, 0) + 1
    keep: set[int] = set()
    for size in (1, 2, 3, 4):
        for pos in range(len(seq) - size + 1):
            gram = tuple(word for _index, word in seq[pos : pos + size])
            if grams[gram] > 1:
                for offset in range(size):
                    keep.add(seq[pos + offset][0])
    return frozenset(keep)


def _anchor_priorities(tokens: tuple[_Token, ...], mode: str) -> list[int | None]:
    repeats = _repeat_indices(tokens)
    result: list[int | None] = []
    for index, tok in enumerate(tokens):
        core = tok.core
        if not core:
            result.append(None)
            continue
        form = _has_form(core)
        capitalized = _is_capitalized(core)
        if mode == "capitalized":
            if not capitalized:
                result.append(None)
            else:
                result.append(_PRIORITY_FORM if form else _PRIORITY_CAPITALIZED)
            continue
        if not _is_content(core):
            result.append(None)
            continue
        if form:
            result.append(_PRIORITY_FORM)
        elif capitalized:
            result.append(_PRIORITY_CAPITALIZED)
        elif core.lower() not in COMMON:
            result.append(_PRIORITY_RARE)
        elif index in repeats:
            result.append(_PRIORITY_REPEAT)
        else:
            result.append(None)
    return result


def _merge_spans(priorities: list[int | None], bridges: list[bool]) -> list[tuple[int, int]]:
    anchors = [index for index, value in enumerate(priorities) if value is not None]
    if not anchors:
        return []
    spans: list[tuple[int, int]] = [(anchors[0], anchors[0])]
    for position in anchors[1:]:
        low, high = spans[-1]
        if position - high - 1 <= _MAX_BRIDGE and all(
            bridges[index] for index in range(high + 1, position)
        ):
            spans[-1] = (low, position)
        else:
            spans.append((position, position))
    return spans


def _extend_span(
    span: tuple[int, int],
    priorities: list[int | None],
    bridges: list[bool],
    capitalized: list[bool],
) -> tuple[int, int]:
    low, high = span
    steps = 0
    while (
        low - 1 >= 0
        and steps < _MAX_BRIDGE
        and priorities[low - 1] is None
        and bridges[low - 1]
        and capitalized[low - 1]
    ):
        low -= 1
        steps += 1
    steps = 0
    while (
        high + 1 < len(priorities)
        and steps < _MAX_BRIDGE
        and priorities[high + 1] is None
        and bridges[high + 1]
        and capitalized[high + 1]
    ):
        high += 1
        steps += 1
    return (low, high)


def _variant_ranges(low: int, high: int) -> list[tuple[int, int]]:
    if high - low + 1 <= _MAX_FULL_VARIANT_SPAN:
        return [(i, j) for i in range(low, high + 1) for j in range(i, high + 1)]
    ranges: list[tuple[int, int]] = []
    for i in range(low, high + 1):
        limit = min(high, i + _MAX_VARIANT_LEN - 1)
        ranges.extend((i, j) for j in range(i, limit + 1))
    ranges.append((low, high))
    return ranges


def cap_frac(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    capitalized = sum(1 for tok in tokens if _is_capitalized(tok.core))
    return capitalized / len(tokens)


def choose_prefilter(text: str) -> str:
    return "capitalized" if cap_frac(text) > 0 else "name_like"


def generate_candidates(text: str, *, mode: str, max_candidates: int = 256) -> list[Candidate]:
    if mode not in ("capitalized", "name_like"):
        raise ValueError(f"unsupported candidate mode: {mode!r}")
    if max_candidates <= 0:
        return []
    tokens = _tokenize(text)
    priorities = _anchor_priorities(tokens, mode)
    bridges = [_is_bridge(tok.core) for tok in tokens]
    capitalized = [_is_capitalized(tok.core) for tok in tokens]
    spans = _merge_spans(priorities, bridges)
    if mode == "name_like":
        spans = [_extend_span(span, priorities, bridges, capitalized) for span in spans]
    best: dict[tuple[int, int], Candidate] = {}
    for low, high in spans:
        for start_index, end_index in _variant_ranges(low, high):
            joined = False
            priority: int | None = None
            for index in range(start_index, end_index + 1):
                value = priorities[index]
                if value is None:
                    joined = True
                elif priority is None or value < priority:
                    priority = value
            if priority is None:
                continue
            if joined:
                priority = _PRIORITY_JOINED
            start = tokens[start_index].start
            end = tokens[end_index].end
            key = (start, end)
            candidate = Candidate(
                text=text[start:end],
                start=start,
                end=end,
                source=_SOURCE_BY_PRIORITY[priority],
                priority=priority,
            )
            previous = best.get(key)
            if previous is None or candidate.priority < previous.priority:
                best[key] = candidate
    result = sorted(best.values(), key=lambda item: (item.start, item.end))
    if len(result) <= max_candidates:
        return result
    ranked = sorted(result, key=lambda item: (item.priority, item.start, item.end))
    kept = ranked[:max_candidates]
    earliest = min(result, key=lambda item: (item.start, item.end))
    if earliest not in kept:
        kept = [*kept[: max_candidates - 1], earliest]
    return sorted(kept, key=lambda item: (item.start, item.end))
