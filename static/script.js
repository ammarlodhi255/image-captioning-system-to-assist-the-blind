const dropArea = document.querySelector(".drag-area")
dragText = dropArea.querySelector("header")
button = dropArea.querySelector("button")
input = dropArea.querySelector("input")
da = document.querySelector(".drag-area").innerHTML
let file; 
removeImageButton = document.getElementById('rmv-img-btn')

function browseBtnCLicked() { 
  document.getElementById('input').click(); 
}


function inputChanged() {
  file = document.getElementById('input').files[0];
  dropArea.classList.add("active");
  showFile(); 
}


function dragOver(event) {
  event.preventDefault(); 
  dropArea.classList.add("active");
  dropArea.querySelector("header").textContent = "Release to Upload File";
}

//If user leaves dragged File from DropArea
function dragLeave() {
  dropArea.classList.remove("active");
  dropArea.querySelector("header").textContent = "Drag & Drop to Upload File";
}


function dropped(event) {
  event.preventDefault(); 

  //getting user selected file and [0] means if user select multiple files then we'll only the first
  file = event.dataTransfer.files[0];
  showFile(); 
}

function showFile() {
  document.getElementById('caption').innerText = ''
  let fileType = file.type; 
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"]; 
  
  if(validExtensions.includes(fileType)) { 
    let fileReader = new FileReader(); 
    fileReader.onload = () => {
      let fileURL = fileReader.result; 
      //creating an img tag and passing user selected file source inside src attribute
      let imgTag = `<img src="${fileURL}" alt="image" class="imageDiv">`; 
      dropArea.innerHTML = imgTag; //adding that created img tag inside dropArea container
    }

    fileReader.readAsDataURL(file);
    removeImageButton.style.display = "block";

  } else {
      alert("This is not an Image File!");
      dropArea.classList.remove("active");
      dragText.textContent = "Drag & Drop to Upload File";
  }
}

function rmvButtonClicked() {
  dropArea.innerHTML = da;
  document.getElementById('caption').innerText = '';
  removeImageButton.style.display = "none";
  file = null
}

function submitFormAndShowCaption() {
  spinner = document.getElementById("loader")
  spinner.style.display = "block";
  document.getElementById("form1").submit(); 
}

function submitPredictAgainForm() {
  document.getElementById("predict-againform").submit();
}

function submitSpeakForm() {
  document.getElementById("speakForm").submit();
}