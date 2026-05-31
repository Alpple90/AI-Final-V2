# map.py - node connections, colours, coordinate loading and edge drawing
import tkinter as tk
import tkintermapview
import pandas as pd
from config import MAP_TILE_SERVER, MAP_ZOOM_LEVEL

TRUE_FILE = "scatsTrueLongLat.xlsx"
 
NODE_CONNECTIONS = {
    '970':  ['3685', '2846'],
    '2000': ['3685', '3682', '3812', '4043'],
    '2200': ['3126', '4063'],
    '2820': ['3662', '4321', '2825'],
    '2825': ['2820', '4030', '2827'],
    '2827': ['2825', '4051'],
    '2846': ['970'],
    '3001': ['4262', '3002', '3662', '4821'],
    '3002': ['4263', '3662', '3001'],
    '3120': ['4040', '3122', '4035'],
    '3122': ['3804', '3127', '3120'],
    '3126': ['3682', '2200', '3127'],
    '3127': ['3126', '4063', '3122'],
    '3180': ['4057', '4051'],
    '3662': ['3001', '3002', '4324', '4335', '2820'],
    '3682': ['2000', '3126', '3804'],
    '3685': ['970',  '2000'],
    '3804': ['3812', '3682', '3122', '4040'],
    '3812': ['2000', '3804', '4040'],
    '4030': ['4321', '4032', '4051', '2825'],
    '4032': ['4034', '4057', '4030', '4321'],
    '4034': ['4035', '4063', '4032', '4324'],
    '4035': ['3120', '4034'],
    '4040': ['4043', '3812', '3804', '3120', '4264', '4272'],
    '4043': ['2000', '4040', '4273'],
    '4051': ['4030', '3180', '2827'],
    '4057': ['4063', '3180', '4032'],
    '4063': ['3127', '2200', '4057', '4034'],
    '4262': ['4263', '3001'],
    '4263': ['4264', '3002', '4262'],
    '4264': ['4270', '4040', '4324', '4263'],
    '4270': ['4272', '4264', '4812'],
    '4272': ['4273', '4040', '4270'],
    '4273': ['4043', '4272'],
    '4321': ['4335', '4032', '4030', '2820'],
    '4324': ['4264', '4034', '3662'],
    '4335': ['3662', '4321'],
    '4812': ['4270'],
    '4821': ['3001'],
}
 
NODE_COLOURS = {
    '970':  '#e63946', '2000': '#457b9d', '2200': '#2a9d8f',
    '2820': '#e9c46a', '2825': '#f4a261', '2827': '#264653',
    '2846': '#8338ec', '3001': '#fb5607', '3002': '#ff006e',
    '3120': '#06d6a0', '3122': '#118ab2', '3126': '#ffd166',
    '3127': '#06a77d', '3180': '#d62828', '3662': '#023e8a',
    '3682': '#80b918', '3685': '#e07a5f', '3804': '#3d405b',
    '3812': '#81b29a', '4030': '#f2cc8f', '4032': '#e76f51',
    '4034': '#a8dadc', '4035': '#457b9d', '4040': '#c77dff',
    '4043': '#48cae4', '4051': '#b5179e', '4057': '#7b2d00',
    '4063': '#4cc9f0', '4262': '#f77f00', '4263': '#d62246',
    '4264': '#4f772d', '4270': '#90e0ef', '4272': '#ff4d6d',
    '4273': '#52b788', '4321': '#c9184a', '4324': '#ff9f1c',
    '4335': '#2ec4b6', '4812': '#e71d36', '4821': '#011627',
}
 
 
# read excel, split lat/long column, strip leading zeros from SCATS numbers
def loadSites():
    df = pd.read_excel(TRUE_FILE)
    df[['LAT', 'LNG']] = df['Lat Long'].str.split(expand=True).astype(float)
    df['SCATS Number'] = df['SCATS Number'].astype(str).str.lstrip('0').str.strip()
    df.loc[df['SCATS Number'] == '', 'SCATS Number'] = '0'
    return df[['SCATS Number', 'LAT', 'LNG']]


# draw each edge once by tracking sorted node pairs
def drawEdges(mapWidget, coords):
    drawn = set()
    paths = []
    for a, neighbours in NODE_CONNECTIONS.items():
        if a not in coords:
            continue
        for b in neighbours:
            if b not in coords:
                continue
            key = tuple(sorted([a, b]))
            if key in drawn:
                continue
            drawn.add(key)
            latA, lngA = coords[a]
            latB, lngB = coords[b]
            p = mapWidget.set_path([(latA, lngA), (latB, lngB)], color='#888888', width=2)
            paths.append(p)
    return paths


if __name__ == '__main__':
    sites = loadSites()
    coords = {r['SCATS Number']: (r['LAT'], r['LNG']) for _, r in sites.iterrows()}

    root = tk.Tk()
    root.title('SCATS Node Graph')
    root.geometry('1200x800')

    mapWidget = tkintermapview.TkinterMapView(root, corner_radius=0)
    mapWidget.pack(fill='both', expand=True)
    mapWidget.set_tile_server(MAP_TILE_SERVER)
    lats = [c[0] for c in coords.values()]
    lngs = [c[1] for c in coords.values()]
    mapWidget.set_position(sum(lats) / len(lats), sum(lngs) / len(lngs))
    mapWidget.set_zoom(MAP_ZOOM_LEVEL)

    drawEdges(mapWidget, coords)

    for _, row in sites.iterrows():
        sid = row['SCATS Number']
        mapWidget.set_marker(
            row['LAT'], row['LNG'],
            text=sid,
            marker_color_circle=NODE_COLOURS.get(sid, '#1a1a2e'),
            marker_color_outside='#ffffff',
        )

    root.mainloop()