preferences.rulerUnits = Units.PIXELS;

const doc = app.activeDocument;
const layer = doc.activeLayer;

const sampler = doc.colorSamplers[0];

var x = sampler.position[0];
var y = sampler.position[1];

x /= doc.width;
y /= doc.height;

const outPath = "C:/Users/aleks/OneDrive - Suomalaisen Yhteiskoulun Osakeyhtiö/Tiedostot/Coding/EE/tickets-ai/samples/";
const filename =  outPath + layer.name + ", x=" + x.toFixed(3) + ", y=" + y.toFixed(3) + ".png";
const pngFile = new File(filename);
const pngOptions = new PNGSaveOptions();
pngOptions.compression = 5;

doc.saveAs(pngFile, pngOptions);

