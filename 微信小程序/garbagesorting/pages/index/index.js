Page({
	data: {
		imgsrc: "",
		state: 1,
		res: '',
		index: 0,
		array: [
			'google/vit-base-patch16-224-in21k',
			"microsoft/swin-large-patch4-window12-384-in22k",
			"microsoft/resnet-50",
			"cnu/resnet 50"]
	},
	onLoad() {

	},
	bindPickerChange(e){
		this.setData({
      index: e.detail.value
    })
	},
	takePhoto() {
		this.setData({
			state: 1
		})
		wx.chooseMedia({
			count: 1,
			mediaType: ['image'],
			sourceType: ['album', 'camera'],
			camera: 'back',
			sizeType: ['compressed'],
			success: (res) => {
				console.log(res)
				this.setData({
					imgsrc: res.tempFiles[0].tempFilePath
				})
				this.setData({
					state: 2
				})
				wx.getFileSystemManager().readFile({
					filePath: res.tempFiles[0].tempFilePath, //选择图片返回的相对路径
					encoding: 'base64', //编码格式
					success: res => { //成功的回调
						wx.request({
							url: 'https://imaginer.fun:90/image',
							method: 'POST',
							data: { // image: 第一位表示方法，后面表示图片的base64值
								image: this.data.index + res.data
							},
							success: (res) => {
								this.setData({
									res: res.data.data
								})
								this.setData({
									state: 3
								})
							},
							fail: (res) =>{
								this.setData({
									state: 4
								})
							}
						})
					}
				})
			}
		})
	}
})