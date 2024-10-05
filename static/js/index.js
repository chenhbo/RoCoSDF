window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 82;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "/" + String(i + 1).padStart(6, "0") + ".jpg";
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = document.getElementById("interpolation-image");
  image.src = interp_images[i].src; // Update the image source
}

$(document).ready(function () {
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    autoplay: false,
    autoplaySpeed: 3000,
  };

  var carousels = bulmaCarousel.attach(".carousel", options);
  for (var i = 0; i < carousels.length; i++) {
    carousels[i].on("before:show", (state) => {
      console.log(state);
    });
  }

  preloadInterpolationImages();

  $("#interpolation-slider").on("input", function (event) {
    setInterpolationImage(this.value);
  });

  setInterpolationImage(0);
  $("#interpolation-slider").prop("max", NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();
});

document.addEventListener('DOMContentLoaded', () => {
  const videos = document.querySelectorAll('video');
  let loadedCount = 0;

  videos.forEach(video => {
    video.autoplay = false; // Disable autoplay initially

    video.addEventListener('loadeddata', () => {
      loadedCount++;
      
      // Check if all videos are loaded
      if (loadedCount === videos.length) {
        videos.forEach(v => v.play());
      }
    });
  });
});
