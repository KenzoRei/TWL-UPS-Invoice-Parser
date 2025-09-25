import pickle
from tkinter import Tk, filedialog

# Hide the main tkinter window
root = Tk()
root.withdraw()

# Let user select a .pkl file
pkl_file = filedialog.askopenfilename(
    title="Select a .pkl file",
    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
)
if not pkl_file:
    print("No file selected. Exiting.")
else:
    search_trk_num = input("Enter tracking number to search: ").strip()
    with open(pkl_file, "rb") as f:
        invoices = pickle.load(f)

    found = False
    for inv in invoices.values() if isinstance(invoices, dict) else invoices:
        for shipment in getattr(inv, "shipments", {}).values():
            for pkg in getattr(shipment, "packages", {}).values():
                if getattr(pkg, "trk_num", None) == search_trk_num:
                    print(f"Found in Invoice: {getattr(inv, 'inv_num', '')}")
                    print(f"Shipment: {getattr(shipment, 'main_trk_num', '')}")
                    print(f"Package Tracking #: {pkg.trk_num}")
                    print(f"Entered Weight: {pkg.entered_wgt}")
                    print(f"Billed Weight: {pkg.billed_wgt}")
                    print(f"Length: {pkg.length}, Width: {pkg.width}, Height: {pkg.height}")
                    print(f"Entered Length: {pkg.entered_length}, Entered Width: {pkg.entered_width}, Entered Height: {pkg.entered_height}")
                    print(f"AP Amount: {pkg.ap_amt}, AR Amount: {pkg.ar_amt}")
                    found = True
    if not found:
        print("Tracking number not found.")