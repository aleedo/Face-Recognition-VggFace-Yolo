$(document).ready(function () {
    $("#show-image-btn").on("click", function () {
        // Get the image URL from the input field
        var imageUrl = $("#image-url").val();

        // Check if the URL is valid
        if (isValidUrl(imageUrl)) {
            // Create an image element and set the source to the URL
            var image = $("<img>").attr("src", imageUrl);

            // Clear the image container and append the new image
            $("#image-container").empty().append(image);
        } else {
            // Display an error message if the URL is invalid
            $("#image-container").empty().append("<div class='error-message'>Invalid URL</div>");
        }
    });

    // Function to check if a string is a valid URL
    function isValidUrl(url) {
        var pattern = /^(ftp|http|https):\/\/[^ "]+$/;
        return pattern.test(url);
    }
});