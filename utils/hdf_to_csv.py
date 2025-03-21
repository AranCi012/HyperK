import pandas as pd
import os
import h5py
from dataclasses import dataclass
import plotly.express as px


@dataclass
class HKHitPmt():
    x: float
    y: float
    z: float
    charge: float
    time: float

@dataclass
class HKTrack:
    nevt: int
    run: int
    PID: int
    id: int
    parent_id: int
    x: float
    y: float
    z: float
    dir_x: float
    dir_y: float
    dir_z: float
    time: float
    mom: float


class HKPandasReader():
    def __init__(self, filename):
        self.pmts = pd.read_hdf(filename, mode="r", key="pmts")
        self.mpmts = pd.read_hdf(filename, mode="r", key="mpmts")
        self.evts =  pd.read_hdf(filename, mode="r", key="evts")
        self.hits =  pd.read_hdf(filename, mode="r", key="hits")
        self.hits2 =  pd.read_hdf(filename, mode="r", key="hits2")
        self.tracks =  pd.read_hdf(filename, mode="r", key="tracks")

    def getEventsList(self):
        return self.evts[['nevt', 'run']].drop_duplicates()

    def getPmtPositionById(self, id: int):
        return self.pmts.loc[id, ['x', 'y', 'z']]

    def getPmtDirectionById(self, id: int):
        return self.pmts.loc[id, ['dir_x', 'dir_y', 'dir_z']]

    def getMPmtPositionById(self, id: int):
        return self.mpmts.loc[id, ['x', 'y', 'z']]

    def getMPmtDirectionById(self, id: int):
        return self.mpmts.loc[id, ['dir_x', 'dir_y', 'dir_z']]
    def getMCTrueTracks(self, nevt: int, run: int):
        return self.tracks[(self.tracks['nevt'] == nevt) & (self.tracks['run'] == run)]

    #use it to get data from MC track with id == 0 (for single particle guns)
    def getMCRootTrackData(self, nevt: int, run: int):
        df: pd.DataFrame = self.tracks[(self.tracks['nevt'] == nevt) & (self.tracks['run'] == run) & (self.tracks['id'] == 0)]
        td = df.iloc[0].to_dict()
        track = HKTrack(**td)
        return track

    def getHitPmtData(self, nevt: int, run: int, ntrigger: int = 0) -> list:
        evt_key  = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits[self.hits['evt_key'] == evt_key].merge(self.pmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']].to_records(index=False)
        return [HKHitPmt(*e) for e in df_result]
    def getHitMPmtData(self, nevt: int, run: int, ntrigger: int = 0) -> list:
        evt_key  = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits2[self.hits2['evt_key'] == evt_key].merge(self.mpmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']].to_records(index=False)
        return [HKHitPmt(*e) for e in df_result]

    def plotEventPmtCharge(self, nevt: int, run: int, ntrigger: int = 0):
        evt_key = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits[self.hits['evt_key'] == evt_key].merge(self.pmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']]
        fig = px.scatter_3d(df_result, x='x', y='y', z='z', color='charge')
        fig.show()

    def plotEventPmtTime(self, nevt: int, run: int, ntrigger: int = 0):
        evt_key = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits[self.hits['evt_key'] == evt_key].merge(self.pmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']]
        fig = px.scatter_3d(df_result, x='x', y='y', z='z', color='time')
        fig.show()

    def plotEventMPmtCharge(self, nevt: int, run: int, ntrigger: int = 0):
        evt_key = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits2[self.hits2['evt_key'] == evt_key].merge(self.mpmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']]
        fig = px.scatter_3d(df_result, x='x', y='y', z='z', color='charge')
        fig.show()

    def plotEventMPmtTime(self, nevt: int, run: int, ntrigger: int = 0):
        evt_key = self.evts.index[(self.evts['nevt'] == nevt) & (self.evts['run'] == run) & (self.evts['ntrigger'] == ntrigger)][0]
        df_result = self.hits2[self.hits2['evt_key'] == evt_key].merge(self.mpmts, how='left', left_on="id", right_index=True)[['x', 'y', 'z', 'charge', 'time']]
        fig = px.scatter_3d(df_result, x='x', y='y', z='z', color='time')
        fig.show()


def export_pmt_hit_track_data_by_event(reader: HKPandasReader, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    events = reader.getEventsList()

    for _, row in events.iterrows():
        nevt, run = row["nevt"], row["run"]

        try:
            mc_track = reader.getMCRootTrackData(nevt, run)

            evt_key = reader.evts.index[
                (reader.evts['nevt'] == nevt) &
                (reader.evts['run'] == run) &
                (reader.evts['ntrigger'] == 0)
            ][0]
            hits_df = reader.hits[reader.hits['evt_key'] == evt_key][['id', 'charge', 'time']].set_index("id")

            # ⬇️ Qui conserviamo l'id come colonna!
            pmts_df = reader.pmts.copy()
            pmts_df.index.name = "id"  # dai un nome all'indice
            pmts_df = pmts_df.reset_index()  # ora 'id' è colonna!

            pmts_df = pmts_df.join(hits_df, on="id", how="left")
            pmts_df["nevt"] = nevt
            pmts_df["run"] = run

            # Campi MC vuoti per PMT
            pmts_df["mc_x"] = None
            pmts_df["mc_y"] = None
            pmts_df["mc_z"] = None
            pmts_df["mc_dir_x"] = None
            pmts_df["mc_dir_y"] = None
            pmts_df["mc_dir_z"] = None

            # Riga MC
            mc_row = {
                "nevt": nevt,
                "run": run,
                "id": -1,
                "x": None,
                "y": None,
                "z": None,
                "dir_x": None,
                "dir_y": None,
                "dir_z": None,
                "charge": None,
                "time": None,
                "mc_x": mc_track.x,
                "mc_y": mc_track.y,
                "mc_z": mc_track.z,
                "mc_dir_x": mc_track.dir_x,
                "mc_dir_y": mc_track.dir_y,
                "mc_dir_z": mc_track.dir_z
            }

            columns = [
                "nevt", "run", "id",
                "x", "y", "z",
                "dir_x", "dir_y", "dir_z",
                "charge", "time",

                "mc_x", "mc_y", "mc_z",
                "mc_dir_x", "mc_dir_y", "mc_dir_z",
            ]

            out_df = pd.concat([pmts_df, pd.DataFrame([mc_row])], ignore_index=True)
            out_df = out_df[columns]

            out_file = os.path.join(output_dir, f"event_nevt{nevt}_run{run}.csv")
            out_df.to_csv(out_file, index=False)
            print(f"[✓] Salvato: {out_file}")
            input()

        except Exception as e:
            print(f"[!] Errore nell'evento (nevt={nevt}, run={run}): {e}")




def main():

    hdf_file = "HyperK/Hyperk/shared/100k_ranvtx_ranmom_0_1000_pandas/out.ext.pandas.0"  

    with h5py.File(hdf_file, "r") as f:
        def print_hdf_structure(name, obj):
            print(f"{name}: {obj}")

        print("\n📂 Struttura del file HDF5:")
        f.visititems(print_hdf_structure)

    with h5py.File(hdf_file, "r") as f:
        def print_dataset_columns(key):
            group = f[key]
            if "block0_items" in group:
                column_names = [c.decode() for c in group["block0_items"][:]]
                print(f"📂 Dataset: {key} → Colonne: {column_names}")

        print("\n📊 Colonne disponibili nei dataset:")

        for dataset_name in f.keys():
            print_dataset_columns(dataset_name)

    reader = HKPandasReader("HyperK/Hyperk/shared/100k_ranvtx_ranmom_0_1000_pandas/out.ext.pandas.0")
    export_pmt_hit_track_data_by_event(reader, output_dir="csv_events")
   

    
if __name__ == '__main__':

    main()
