let model = null;

(async () => {
  console.log("Cargando model...");
  model = await tf.loadLayersModel("./model/model.json");
  console.log("model cargado...");
})();

function changeCelsius() {
  let celsius = document.getElementById("celsius").value;
  if (model != null) {
    let tensor = tf.tensor1d([parseInt(celsius)]);
    let fahrenheit = model.predict(tensor).dataSync();
    fahrenheit = Math.round(fahrenheit * 100) / 100;
    document.getElementById(
      "resultado"
    ).innerHTML = `${celsius}º Celsius son ${fahrenheit}º Fahrenheit`;
  } else {
    document.getElementById("resultado").innerHTML = "Cargando...";
  }
}
