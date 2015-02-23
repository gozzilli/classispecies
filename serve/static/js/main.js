
function toggleAll() {
    $('.dir').toggle("fast");
}

function loadPlot(elem, path) {
    $(elem).html("<img src='"+path+"' />").toggle("fast");
}
