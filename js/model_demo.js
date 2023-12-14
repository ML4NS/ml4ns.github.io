import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.11.0';
env.allowLocalModels = false;


// Initialise worker
const worker = new Worker(new URL('./worker.js', import.meta.url), {
    type: 'module',
});


const PROGRESS = document.getElementById('progress');
const PROGRESS_BARS = document.getElementById('progress-bars');


// get input text
let inputText = document.getElementById('inputText');
// get output div
let outputDiv = document.getElementById('model_output');
// get submit button
let submitButton = document.getElementById('submitButton');
// get status
let status = document.getElementById("status");





// on submit button click
submitButton.addEventListener('click', async (e) => {
    let data = {
        task: "sequence-classification",
        text: inputText.value,
        elementIdToUpdate: outputDiv.id,
    };

    // prevent default form submission
    e.preventDefault();

    // Send data to worker
    console.log("Sending data to worker: ", data)

    worker.postMessage(data);
});



worker.addEventListener('message', (event) => {
    const message = event.data;

    switch (message.type) {
        case 'download':

            if (message.data.status === 'initiate') {
                PROGRESS.style.display = 'block';

                // create progress bar
                PROGRESS_BARS.appendChild(htmlToElement(`
                        <div class="progress" model="${message.data.name}" file="${message.data.file}">
                            <div class="progress-bar" role="progressbar"></div>
                        </div>
                    `));

            } else {
                let bar = PROGRESS_BARS.querySelector(`.progress[model="${message.data.name}"][file="${message.data.file}"]> .progress-bar`)

                switch (message.data.status) {
                    case 'progress':
                        // update existing bar
                        bar.style.width = message.data.progress.toFixed(2) + '%';
                        bar.textContent = `${message.data.file} (${formatBytes(message.data.loaded)} / ${formatBytes(message.data.total)})`;
                        break;

                    case 'done':
                        // Remove the progress bar
                        bar.parentElement.remove();
                        break;

                    case 'ready':
                        // Pipeline is ready - hide container
                        PROGRESS.style.display = 'none';
                        PROGRESS_BARS.innerHTML = '';
                        break;
                }
            }

            break;

        case 'complete':

            let result = message.data
            let output_scores = "<br/>"

            result = result.sort(function (a, b) {
                var textA = a.label.toUpperCase();
                var textB = b.label.toUpperCase();
                return (textA > textB) ? -1 : (textA < textB) ? 1 : 0;
            }
            );

            result.forEach(element => {
                let label = element.label
                let score = Math.round(element.score * 100)
                console.log(label, " : ", score, "%", "\n")

                output_scores = output_scores.concat(label, " : ", score, "%", "<br/>")

            });

            console.log(result)
            console.log(output_scores)
            // showing the result
            console.log("Updating element: ", message.target)
            outputDiv = document.getElementById(message.target);
            outputDiv.innerHTML = "Output:\n".concat(output_scores);

            break;

        default:
            break;
    }
});


function htmlToElement(html) {
    // https://stackoverflow.com/a/35385518
    let template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}


function formatBytes(bytes, decimals = 0) {
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    if (bytes === 0) return "0 Bytes";
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1000)), 10);
    const rounded = (bytes / Math.pow(1000, i)).toFixed(decimals);
    return rounded + " " + sizes[i];
}

/*
let output_scores = "<br/>"

result = result.sort(function (a, b) {
    var textA = a.label.toUpperCase();
    var textB = b.label.toUpperCase();
    return (textA > textB) ? -1 : (textA < textB) ? 1 : 0;
}
);

result.forEach(element => {
    let label = element.label
    let score = Math.round(element.score * 100)
    console.log(label, " : ", score, "%", "\n")

    output_scores = output_scores.concat(label, " : ", score, "%", "<br/>")

});

console.log(result)
console.log(output_scores)
// showing the result
outputDiv = document.getElementById(data.elementIdToUpdate);
outputDiv.innerHTML = "Output:\n".concat(output_scores);
*/


/*
// on submit button click
submitButton.addEventListener('click', async (e) => {

    status.textContent = "Loading model, this will take some time on the first run...";

    // prevent default form submission
    e.preventDefault();

    // Specifying classifier task
    let classifier = await pipeline(
        'sentiment-analysis',
        'Xenova/bert-base-multilingual-uncased-sentiment',
    );

    status.textContent = "Calculating output...";

    // classifying the input text
    let result = await classifier(
        inputText.value, { topk: 5 }
    );

    status.textContent = "";

    let output_scores = "<br/>"

    result = result.sort(function (a, b) {
        var textA = a.label.toUpperCase();
        var textB = b.label.toUpperCase();
        return (textA > textB) ? -1 : (textA < textB) ? 1 : 0;
    }
    );

    result.forEach(element => {
        let label = element.label
        let score = Math.round(element.score * 100)
        console.log(label, " : ", score, "%", "\n")

        output_scores = output_scores.concat(label, " : ", score, "%", "<br/>")

    });

    console.log(result)
    console.log(output_scores)
    // showing the result
    outputDiv.innerHTML = "Output:\n".concat(output_scores);
});
*/ 