class Slider {
    constructor (rangeElement, valueElement, options) {
      this.rangeElement = rangeElement
      this.valueElement = valueElement
      this.options = options
  
      // Attach a listener to "change" event
      this.rangeElement.addEventListener('input', this.updateSlider.bind(this))
    }
  
    // Initialize the slider
    init() {
      this.rangeElement.setAttribute('min', options.min)
      this.rangeElement.setAttribute('max', options.max)
      this.rangeElement.value = options.cur
  
      this.updateSlider()
    }
  
    // Format the money
    format(value) {
      return parseFloat(value)
        .toLocaleString('en-US', { maximumFractionDigits: 1 })
    }
  
    generateBackground(rangeElement) {   
      if (this.rangeElement.value === this.options.min) {
        return
      }
  
      let percentage =  (this.rangeElement.value - this.options.min) / (this.options.max - this.options.min) * 100
      return 'background: linear-gradient(to right, #50299c, #7a00ff ' + percentage + '%, #d3edff ' + percentage + '%, #dee1e2 100%)'
    }
  
    updateSlider (newValue) {
      this.valueElement.innerHTML = this.format(this.rangeElement.value)
      this.rangeElement.style = this.generateBackground(this.rangeElement.value)
    }
  }
  
  let rangeElement = document.querySelector('.range [type="range"]')
  let valueElement = document.querySelector('.range .range__value span') 
  
  let options = {
    min: 2,
    max: 5,
    cur: 2
  }
  
  if (rangeElement) {
    let slider = new Slider(rangeElement, valueElement, options)
    slider.init()
    sendData(slider)
  }

  function sendData(slider) {
    $.ajax({
        url: '/process',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 'value': slider }),
        success: function(response) {
            document.getElementById('output').innerHTML = response.result;
        },
        error: function(error) {
            console.log(error);
        }
    });
}
