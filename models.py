class Location:
    def __init__(self, company: str = "", contact: str = "", addr1: str = "", addr2: str = "",
                 city: str = "", state: str = "", zipcode: str = "", country: str = "", 
                 contact_num: str = "", contact_email: str = ""):
        self.company = company
        self.contact = contact
        self.addr1 = addr1
        self.addr2 = addr2
        self.city = city
        self.state = state
        self.zipcode = zipcode
        self.country = country
        self.contact_num = contact_num
        self.contact_email = contact_email

class Charge:
    def __init__(self, charge_en: str = "", charge_cn: str = "", 
                 inc_amt: float = 0.0, ap_amt: float = 0.0, ar_amt: float = 0.0, charge_ref1: str = "", 
                 charge_ref2: str = ""):
        self.charge_en = charge_en
        self.charge_cn = charge_cn
        self.inc_amt = inc_amt
        self.ap_amt = ap_amt
        self.ar_amt = ar_amt
        self.charge_ref1 = charge_ref1
        self.charge_ref2 = charge_ref2

class Package:
    def __init__(self, trk_num: str = "", entered_wgt: float = 0.0, 
                 billed_wgt: float = 0.0, length: float = 0.0, width: float = 0.0, height: float = 0.0,
                 ap_amt: float = 0.0, ar_amt: float = 0.0, flag_UPS_SCC: bool = False,
                 pkg_ref1: str = "", pkg_ref2: str = ""):
        self.trk_num = trk_num
        self.entered_wgt = entered_wgt
        self.billed_wgt = billed_wgt
        self.length = length
        self.width = width
        self.height = height
        self.entered_length = ""
        self.entered_width = ""
        self.entered_height = ""
        self.ap_amt = ap_amt
        self.ar_amt = ar_amt
        self.charge_detail: dict[str, Charge] = {}
        self.flag_UPS_SCC = flag_UPS_SCC
        self.pkg_ref1 = pkg_ref1
        self.pkg_ref2 = pkg_ref2

from datetime import date
class Shipment:
    def __init__(self, main_trk_num: str = "", pkg_qty: int = 0, cust_id: str = "", 
                 ship_date: date = None, deli_date: date = None, tran_date: date = None, 
                 entered_wgt: float = 0.0, billed_wgt: float = 0.0, zone: str = "", 
                 ap_amt: float = 0.0, ar_amt: float = 0.0, ship_ref1: str = "", ship_ref2: str = ""):
        self.main_trk_num = main_trk_num
        self.pkg_qty = pkg_qty
        self.cust_id = cust_id
        self.ship_date = ship_date
        self.deli_date = deli_date
        self.tran_date = tran_date
        self.entered_wgt = entered_wgt
        self.billed_wgt = billed_wgt
        self.zone = zone
        self.ap_amt = ap_amt
        self.ar_amt = ar_amt
        self.sender: Location = Location()
        self.consignee: Location = Location()
        self.ship_ref1 = ship_ref1
        self.ship_ref2 = ship_ref2
        self.packages: dict[str, Package] = {}
        self.shipment_charge: dict[str, Charge] = {}

class Invoice:
    def __init__(self, carrier: str = "", inv_date: date = None, inv_num: str = "",
                 acct_num: str = "", batch_num : str = "", ap_amt: float = 0.0, 
                 ar_amt: float = 0.0):
        self.carrier = carrier
        self.inv_date = inv_date
        self.inv_num = inv_num
        self.acct_num = acct_num
        self.batch_num = batch_num
        self.ap_amt = ap_amt
        self.ar_amt = ar_amt
        self.inv_charge: dict[str, Charge] = {}
        self.shipments: dict[str, Shipment] = {}