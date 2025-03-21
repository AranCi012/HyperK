import pandas
import pandas as pd
from dataclasses import dataclass
import plotly.express as px

# IMPORTANT NOTE !!!
# This is just an example, please improve it according to your needs !!!

#add direction if needed
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

def main():
    reader = HKPandasReader("example/pandas.out")
    #reader.plotEventMPmtCharge(2, 0)
    events = reader.getEventsList()
    for index, row in events.iterrows():
        reader.getMCRootTrackData(nevt=row["nevt"], run=row["run"])

if __name__ == '__main__':
    # inFileName = sys.argv[1]
    main()


