
document.addEventListener('DOMContentLoaded', function () {
	//////////////
	//////////////
	document.querySelector('#btn-print').addEventListener('click', function () {
		// window.print();

		let wspFrame = document.getElementById('wrapper').contentWindow;
		wspFrame.focus();
		wspFrame.print();
	});
	//////////////
	//////////////

	document.querySelector('#btn-one').addEventListener('click', function () {
		html2canvas(document.querySelector('#graphs')).then((canvas) => {
			let base64image = canvas.toDataURL('image/png');
			// console.log(base64image);
			let pdf = new jsPDF('p', 'px', [1600, 1131]);
			pdf.addImage(base64image, 'PNG', 15, 15, 1110, 360);
			pdf.save('webtylepress-two.pdf');
		});
	});
	
});