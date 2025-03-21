import pandas
import pandas as pd
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


class HKPandasReaderToCsv:
    def __init__(self, filename):
        self.pmts = pd.read_hdf(filename, mode="r", key="pmts")
        self.evts = pd.read_hdf(filename, mode="r", key="evts")
        self.hits = pd.read_hdf(filename, mode="r", key="hits")
        self.tracks = pd.read_hdf(filename, mode="r", key="tracks")

    def getMCTrueTracks(self, nevt: int, run: int) -> pd.DataFrame:
        return self.tracks[(self.tracks['nevt'] == nevt) & (self.tracks['run'] == run)]

    def getPmtDirectionById(self, id: int):
        # Recupera la direzione di un PMT dato il suo ID
        return self.pmts.loc[id, ['dir_x', 'dir_y', 'dir_z']]

    def getHitPmtData(self, nevt: int, run: int, ntrigger: int = 0) -> pd.DataFrame:
        # Recupera l'indice dell'evento corrispondente
        evt_key = self.evts.index[
            (self.evts['nevt'] == nevt) & 
            (self.evts['run'] == run) & 
            (self.evts['ntrigger'] == ntrigger)
        ][0]
        
        # Filtra gli hit per l'evento specifico e combina i dati con i PMT
        df_hits = self.hits[self.hits['evt_key'] == evt_key].merge(
            self.pmts, how='left', left_on="id", right_index=True
        )[['id','x', 'y', 'z', 'dir_x', 'dir_y', 'dir_z','charge', 'time']]
        df_tracks = self.getMCTrueTracks(nevt, run)
        df_tracks = df_tracks.assign(id=-1)
        df_final = pd.concat([df_hits, df_tracks], ignore_index=True)
        df_final = df_final.sort_values(by="id", ignore_index=True)
        
        return df_final
      

    def saveEventsToCsv(self, output_folder: str):
        # Itera su tutti gli eventi per salvarli in file CSV
        events = self.evts[['nevt', 'run']].drop_duplicates()
        for _, row in events.iterrows():
            nevt = row['nevt']
            run = row['run']
            
            try:
                # Recupera i dati per l'evento specifico
                df_event = self.getHitPmtData(nevt, run)
                # Salva il risultato in un file CSV
                filename = f"{output_folder}/event_{nevt}_run_{run}.csv"
                df_event.to_csv(filename, index=False)
                print(f"Salvato: {filename}")
            except Exception as e:
                print(f"Errore con evento {nevt}, run {run}: {e}")




def main():
    # Specifica il percorso del file HDF5 e della cartella di output
    input_file = "Hyperk/hk/out.ext.pandas.0"
    output_folder = "Hyperk/hk/output_events"
    
    # Crea l'oggetto reader
    reader = HKPandasReaderToCsv(input_file)
    
    # Salva gli eventi in file CSV
    reader.saveEventsToCsv(output_folder)

if __name__ == '__main__':
    main()
