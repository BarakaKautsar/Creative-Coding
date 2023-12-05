// Create a variable to hold our map
let myMap;
// Create a variable to hold our canvas
let canvas;
// Create a new Mappa instance using Leaflet.
let mappa = new Mappa("Leaflet");
let data = [];

// Lets put all our map options in a single object
const options = {
  lat: 0,
  lng: 0,
  zoom: 2,
  style: "http://{s}.tile.osm.org/{z}/{x}/{y}.png",
};

function preload() {
  annualData = loadTable("data/annual_data.csv", "header");
  countries = loadTable("data/country_data.csv", "header");
}
function setup() {
  canvas = createCanvas(640, 640);
  // background(100); let's uncomment this, we don't need it for now

  // Create a tile map with the options declared
  myMap = mappa.tileMap(options);
  myMap.overlay(canvas);
}

function draw() {
  clear();
  for (let row of annualData.rows) {
    let country = row.get("Entity");
    let coord = countries.findRow(country, "name");
    if (coord === null) {
      continue;
    }
    let lat = coord.get("latitude");
    let lon = coord.get("longitude");
    const pix = myMap.latLngToPixel(lat, lon);
    size = map(row.get("Annual CO₂ emissions"), 0, 1000000000, 1, 10);
    console.log(country);
    // fill(frameCount % 255, 0, 200, 100);
    // const zoom = myMap.zoom();
    // const scl = pow(2, zoom); // * sin(frameCount * 0.1);
    ellipse(pix.x, pix.y, size);
  }
}
