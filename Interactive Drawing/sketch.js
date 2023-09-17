let shapes = []; // making array of shapes

class Shape {
  // making class of shapes
  constructor(x, y, r) {
    this.x = x;
    this.y = y;
    this.r = r; // radius
    this.speed = createVector(0, 0); // set initial speed to 0 or stationary
    this.color = color(random(255), random(255), random(255)); // random color
  }

  display() {
    noStroke();
    fill(this.color);
    circle(this.x, this.y, this.r);
  }

  move() {
    //changing position with speed vector
    this.x += this.speed.x;
    this.y += this.speed.y;

    // bouncing on the edges
    if (this.x + this.r / 2 > width || this.x - this.r / 2 < 0) {
      this.speed.x *= -1;
    }

    if (this.y + this.r / 2 > height || this.y - this.r / 2 < 0) {
      this.speed.y *= -1;
    }
  }

  insideShape(x, y) {
    //check if the point (x,y) is inside shape
    let d = dist(x, y, this.x, this.y);
    return d < this.r / 2;
  }

  changeSpeed() {
    // this.color = color(random(255), random(255), random(255));
    if (this.speed.x == 0 && this.speed.y == 0) {
      this.speed = createVector(random(-3, 3), random(-3, 3));
    } else {
      this.speed = createVector(0, 0);
    }
  }
}

function setup() {
  createCanvas(400, 400);
  for (let i = 0; i < 5; i++) {
    // making 5 shapes
    let shape = new Shape(random(width), random(height), random(50, 70));
    shapes.push(shape);
  }
}

function draw() {
  background(250);

  for (let i = 0; i < shapes.length; i++) {
    // display and move all shapes
    shapes[i].display();
    shapes[i].move();
  }
}

function mouseClicked() {
  // when mouse is clicked, check if it is inside shape and change speed
  for (let i = 0; i < shapes.length; i++) {
    if (shapes[i].insideShape(mouseX, mouseY)) {
      shapes[i].changeSpeed();
    }
  }
}
