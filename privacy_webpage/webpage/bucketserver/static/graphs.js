function append_plotly_plot_dict(container, plot_dict){
	var div = document.createElement("div");
    div.style.display = "block";
    div.style.width = "700px";
    div.style.height = "320px";
    container.appendChild(div);

    Plotly.plot(div, plot_dict['traces'], plot_dict['layout']);
}

// this function deploys plotly plots in the given container
function create_plotly_plots(container, plots_dict_string){
	plots_dict = JSON.parse(plots_dict_string);
	// The data dict should be ordered
	for (const [key, value] of Object.entries(plots_dict)) {
  		console.log("elaborate on plot-key: " + key);
  		append_plotly_plot_dict(container, value);
	}
}


// this functions displays a base64 encoded image in the given container
function create_base64string_plot(container, string){
	var img = document.createElement("img");
    img.style.display = "block";
    img.style.width = "500px";
    img.src = string;
    image_container.appendChild(img);
}

