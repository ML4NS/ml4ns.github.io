import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.11.0';
env.allowLocalModels = false;




// Define task function mapping
const TASK_FUNCTION_MAPPING = {
    'sequence-classification': sequence_classification,
}



// Listen for messages from UI
self.addEventListener('message', async (event) => {
    const data = event.data;
    let fn = TASK_FUNCTION_MAPPING[data.task];

    console.log("worker.js: ", data.task)
    console.log("worker.js: ", data.text)
    console.log("worker.js: ", data.elementIdToUpdate)

    if (!fn) return;

    let result = await fn(data);
    self.postMessage({
        task: data.task,
        type: 'result',
        data: result,
        elementIdToUpdate: data.elementIdToUpdate,
    });
});


// Define model factories
// Ensures only one model is created of each type
class PipelineFactory {
    static task = null;
    static model = null;

    // NOTE: instance stores a promise that resolves to the pipeline
    static instance = null;

    constructor(tokenizer, model) {
        this.tokenizer = tokenizer;
        this.model = model;
    }

    /**
     * Get pipeline instance
     * @param {*} progressCallback 
     * @returns {Promise}
     */
    static getInstance(progressCallback = null) {
        if (this.task === null || this.model === null) {
            throw Error("Must set task and model")
        }
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                progress_callback: progressCallback
            });
        }

        return this.instance;
    }
}


class SequenceClassificationPipelineFactory extends PipelineFactory {
    static task = 'text-classification';
    static model = 'Xenova/bert-base-multilingual-uncased-sentiment';
}




async function sequence_classification(data) {

    let pipeline = await SequenceClassificationPipelineFactory.getInstance(data => {
        self.postMessage({
            type: 'download',
            task: 'sequence-classification',
            data: data
        });
    });

    let outputs = await pipeline(data.text, {
        topk: 5 // return all
    })

    self.postMessage({
        type: 'complete',
        target: data.elementIdToUpdate,
        data: outputs
    });

}