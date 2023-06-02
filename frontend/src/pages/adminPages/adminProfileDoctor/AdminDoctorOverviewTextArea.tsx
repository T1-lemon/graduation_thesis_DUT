import React from 'react'

type Props = {
  onChangeTextArea: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
  des?: string
  className?: string
  rows: number
  title: string
}

const AdminDoctorOverviewTextArea = (props: Props) => {
  const { onChangeTextArea, des, rows, title } = props
  return (
    <div className='overview__div--shortdes'>
      <p className='overview__p--title'>{title}</p>
      <textarea
        cols={30}
        rows={rows}
        onChange={onChangeTextArea}
        defaultValue={des}
        className='overview__textarea'
      />
    </div>
  )
}

export default AdminDoctorOverviewTextArea
