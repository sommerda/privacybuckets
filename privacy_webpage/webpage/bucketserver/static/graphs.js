// this function deploys plotly plots in the given container
function create_plotly_plots(container, plot_dicts_json){
	plot_dicts = JSON.parse(plot_dicts_json);
	
	for (const [key, plot_dict] of Object.entries(plot_dicts)) {

  		var div = document.createElement("div");
		div.classList.add("plotlyplot");
    	container.appendChild(div);

    	Plotly.plot(div, plot_dict['traces'], plot_dict['layout']);
	}
}


// this functions displays a base64 encoded image in the given container
function create_base64string_plot(container, string){
	var img = document.createElement("img");
	img.classList.add("base64stringplot");
    img.src = string;
    image_container.appendChild(img);
}

