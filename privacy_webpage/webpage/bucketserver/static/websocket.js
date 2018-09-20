// written by David Sommer (david.sommer at inf.ethz.ch)

var websocket;
var wsUri = "ws://" + window.location.host + "/analysis";
var console_window;
var has_submitted = false;

// initial page setup, called once page is loaded
function page_setup(){
  // make no_js message invisible
  var no_js_window = document.getElementById("no_js_window");
  no_js_window.style.display = "none";

  // make form visible
  var form_window = document.getElementById("form_window");
  form_window.style.display = "block";

  // setup submit button
  var form = document.getElementById("request_form");
  form.addEventListener("submit", onSubmit);

  // setup global variables for later
  console_window = document.getElementById("console_window");

};

// called when submit button is pressed
function onSubmit(evt){
  if (evt.preventDefault) evt.preventDefault();
  if (! has_submitted){ // only submit once!
    websocket = new WebSocket(wsUri);
    websocket.onopen = ws_onOpen;
    websocket.onmessage = ws_onMessage;
    has_submitted = true;
  }
  return false;
}

// called when websocket is ready
function ws_onOpen(evt){
  send_form_data(document.querySelector("form"), websocket);

  // make form invisible
  var form_window = document.getElementById("form_window");
  form_window.style.display = "none";

  // make result visible
  var result_window = document.getElementById("result_window");
  result_window.style.display = "block";
}

// called if message from server received
function ws_onMessage(evt)
{
  msg = JSON.parse(evt.data);
  if (msg.type == 'console'){
    console_window.innerHTML += msg.data;
    console_window.scrollTop = console_window.scrollHeight;
  }
  else if (msg.type == 'image_1'){
    var img = document.createElement("img");
    img.style.display = "block";
    img.style.width = "500px";
    img.src = msg.data;
    document.getElementById("image_container").appendChild(img);

    // improper way to make the results visible:
    document.getElementById("finite_result").style.display = "block";
  }
  else{
    console.log("Received Msg, type " + msg.type);
  }
}

// sends form data via web socket
function send_form_data(form, websocket){
  formData = new FormData(form);

  var ret = {};
  for (pair of formData.entries()){
    key = pair[0];
    value = pair[1];

    // handle file objects asynchronously
    if (value instanceof File){ 
      ret[key] = key;
      
      var reader = new FileReader(value);
      reader.key = key;
      reader.onload = function() {
        var arrayBuffer = this.result; 
        var array = new Uint8Array(arrayBuffer);
        var binaryString = new TextDecoder("utf-8").decode(array);
        //var binaryString = String.fromCharCode.apply(null, array);
        console.log(binaryString);
        console.log(this.key);

        var data = { 'name': this.key, 'data' : binaryString };
        
        websocket.send( JSON.stringify( { 'type': 'file', 'data': data } ) );
      };
      reader.readAsArrayBuffer(value);
    }
    else {
      ret[key] = value;
    }
  }

  // send everything except files
  websocket.send( JSON.stringify( { 'type' : 'parameters', 'data': ret } ) ); 
}

window.addEventListener("load", page_setup, false);
